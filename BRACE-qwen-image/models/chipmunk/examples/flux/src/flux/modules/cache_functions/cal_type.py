from .force_scheduler import force_scheduler


def cal_type(cache_dic, current):
    """
    Determine calculation type for this step
    """
    # 🔥 添加调试信息：在step 0-2时打印配置信息
    # if current['step'] <= 2:
    #     debug_info = (
    #         f"[CACHE DEBUG] Step {current['step']}: "
    #         f"fresh_ratio={cache_dic['fresh_ratio']}, "
    #         f"taylor_cache={cache_dic['taylor_cache']}, "
    #         f"fresh_threshold={cache_dic['fresh_threshold']}, "
    #         f"first_enhance={cache_dic.get('first_enhance', 'N/A')}"
    #     )
    #     print(debug_info)

    # 🔥 新增：collect 模式 - 专门用于特征收集
    if cache_dic.get("collect_mode", False):
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: 选择 COLLECT 模式 -> type='full' (特征收集)")
        return

    # 硬性守卫：original 模式无论 interval/first_enhance 如何设置，都应每步 full
    if cache_dic.get("mode") == "original":
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        return

    # 🔥 修复：在 original 模式下，所有步骤都应该是 'full' 类型
    if (
        (cache_dic["fresh_ratio"] == 0.0)
        and (not cache_dic["taylor_cache"])
        and (cache_dic["fresh_threshold"] == 1)
    ):
        # Original mode: 每一步都进行完整计算，无缓存
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: 选择 ORIGINAL 模式 -> type='full'")
        return

    # 🔥 修复：正确实现first_enhance逻辑
    # first_enhance期间的所有步骤都应该是full模式
    in_first_enhance_period = current["step"] < cache_dic["first_enhance"]

    if (cache_dic["fresh_ratio"] == 0.0) and (not cache_dic["taylor_cache"]):
        # FORA:Uniform
        first_step = current["step"] == 0
    else:
        # ToCa: First enhanced - 前first_enhance步都是full模式
        first_step = in_first_enhance_period

    force_fresh = cache_dic["force_fresh"]
    if not first_step:
        fresh_interval = cache_dic["cal_threshold"]
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if (first_step) or (cache_dic["cache_counter"] == fresh_interval - 1):
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        # current['activated_times'].append(current['t'])
        force_scheduler(cache_dic, current)
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: first_step={first_step}, 选择 -> type='full'")

    elif cache_dic["taylor_cache"]:
        cache_dic["cache_counter"] += 1
        if cache_dic.get("cluster_num", 0) > 0:
            current["type"] = "ClusCa"
        else:
            current["type"] = "taylor_cache"
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: 选择 TAYLOR_CACHE 模式 -> type='taylor_cache'")

    elif cache_dic["cache_counter"] % 2 == 1:  # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    # 'cache_noise' 'ToCa' 'FORA'
    elif cache_dic["Delta-DiT"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
        # if current['step'] < 25:
        #    current['type'] = 'FORA'
        # else:
        #    current['type'] = 'aggressive'


######################################################################
# if (current['step'] in [3,2,1,0]):
#    current['type'] = 'full'
