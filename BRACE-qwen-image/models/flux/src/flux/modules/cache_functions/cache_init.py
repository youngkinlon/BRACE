def cache_init(
    timesteps,
    model_kwargs=None,
    mode="Taylor",
    interval=None,
    max_order=None,
    first_enhance=None,
    hicache_scale=None,
):
    """
    Initialization for cache.

    :param timesteps: 时间步序列
    :param model_kwargs: 模型参数（暂未使用）
    :param mode: 缓存模式 ('original', 'ToCa', 'Taylor', 'Taylor-Scaled', 'HiCache', 'HiCache-Analytic', 'Delta', 'collect', 'ClusCa', 'Hi-ClusCa')
    :param interval: 缓存刷新间隔 (覆盖默认值)
    :param max_order: 最大阶数 (覆盖默认值)
    :param first_enhance: 初始增强步数 (覆盖默认值)
    :param hicache_scale: HiCache多项式缩放因子 (覆盖默认值)
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index["layer_index"] = {}
    cache_dic["attn_map"] = {}
    cache_dic["attn_map"][-1] = {}
    cache_dic["attn_map"][-1]["double_stream"] = {}
    cache_dic["attn_map"][-1]["single_stream"] = {}

    cache_dic["k-norm"] = {}
    cache_dic["k-norm"][-1] = {}
    cache_dic["k-norm"][-1]["double_stream"] = {}
    cache_dic["k-norm"][-1]["single_stream"] = {}

    cache_dic["v-norm"] = {}
    cache_dic["v-norm"][-1] = {}
    cache_dic["v-norm"][-1]["double_stream"] = {}
    cache_dic["v-norm"][-1]["single_stream"] = {}

    cache_dic["cross_attn_map"] = {}
    cache_dic["cross_attn_map"][-1] = {}
    cache[-1]["double_stream"] = {}
    cache[-1]["single_stream"] = {}
    cache_dic["cache_counter"] = 0

    for j in range(19):
        cache[-1]["double_stream"][j] = {}
        cache_index[-1][j] = {}
        cache_dic["attn_map"][-1]["double_stream"][j] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["total"] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["img_mlp"] = {}

        cache_dic["k-norm"][-1]["double_stream"][j] = {}
        cache_dic["k-norm"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["k-norm"][-1]["double_stream"][j]["img_mlp"] = {}

        cache_dic["v-norm"][-1]["double_stream"][j] = {}
        cache_dic["v-norm"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["v-norm"][-1]["double_stream"][j]["img_mlp"] = {}

    for j in range(38):
        cache[-1]["single_stream"][j + 19] = {}  # 单流块缓存也使用全局层索引
        cache_index[-1][j + 19] = {}  # 单流块从层19开始编号
        cache_dic["attn_map"][-1]["single_stream"][j] = {}
        cache_dic["attn_map"][-1]["single_stream"][j]["total"] = {}

        cache_dic["k-norm"][-1]["single_stream"][j] = {}
        cache_dic["k-norm"][-1]["single_stream"][j]["total"] = {}

        cache_dic["v-norm"][-1]["single_stream"][j] = {}
        cache_dic["v-norm"][-1]["single_stream"][j]["total"] = {}

    cache_dic["taylor_cache"] = False
    cache_dic["Delta-DiT"] = False
    cache_dic["use_hicache"] = False
    cache_dic["mode"] = mode

    if mode == "original":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "ToCa":
        cache_dic["cache_type"] = "attention"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "Taylor":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 2
        cache_dic["first_enhance"] = 3

    elif mode == "Taylor-Scaled":
        # Taylor + 双重缩放（用于消融对比）
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 2
        cache_dic["first_enhance"] = 3
        cache_dic["prediction_mode"] = "taylor_scaled"
        # 允许通过 model_kwargs 或外部参数传入缩放因子
        model_kwargs = model_kwargs or {}
        cache_dic["hicache_scale_factor"] = model_kwargs.get(
            "hicache_scale_factor", model_kwargs.get("hicache_scale", 0.5)
        )

    elif mode == "HiCache":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["use_hicache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        cache_dic["hicache_scale_factor"] = 0.5  # HiCache 多项式缩放因子

    elif mode == "HiCache-Analytic":
        # Analytic Adaptive Sigma version of HiCache
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["use_hicache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        
        # Enable Analytic Sigma
        cache_dic["analytic_sigma"] = True
        cache_dic["analytic_stats"] = {} # Initialize stats storage
        
        # Config
        model_kwargs = model_kwargs or {}
        cache_dic["analytic_sigma_config"] = {
            # Default alpha≈1.28 gives initial sigma≈0.9 when q≈1 (raw = alpha / sqrt(2))
            "alpha": model_kwargs.get("analytic_sigma_alpha", 1.28),
            "sigma_max": model_kwargs.get("analytic_sigma_max", 1.0),
            "beta": model_kwargs.get("analytic_sigma_beta", 0.01),
            "eps": model_kwargs.get("analytic_sigma_eps", 1e-6),
            # 选填：更稳的统计与更平滑的 sigma
            "q_quantile": model_kwargs.get("analytic_sigma_q_quantile"),  # e.g. 0.95
            "sigma_smooth": model_kwargs.get("analytic_sigma_smooth", 0.0),  # gamma in log sigma EMA
        }

    elif mode == "Delta":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 3
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["Delta-DiT"] = True
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1

    elif mode == "collect":
        # 🔥 新增：专门用于特征收集的模式
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "collect"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1  # 每步都刷新
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = False  # 不使用Taylor缓存
        cache_dic["max_order"] = 0  # 不需要高阶展开
        cache_dic["first_enhance"] = len(timesteps)  # 所有步骤都是enhanced
        cache_dic["collect_mode"] = True  # 标记为收集模式
        # 自动启用特征收集
        cache_dic["enable_feature_collection"] = True

    elif mode == "ClusCa":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        cache_dic["cluster_info"] = {}
        cache_dic["cluster_num"] = 16
        cache_dic["cluster_method"] = "kmeans"
        cache_dic["k"] = 1
        cache_dic["propagation_ratio"] = 0.005
        cache_dic["prediction_mode"] = "taylor"  # ClusCa uses Taylor predictor

        cluster_info_dict = {}
        cluster_info_dict["cluster_indices"] = None
        cluster_info_dict["centroids"] = None

        cache_dic["cluster_info"]["double_stream"] = {}
        cache_dic["cluster_info"]["single_stream"] = {}
        cache_dic["cluster_info"]["double_stream"]["img_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["double_stream"]["txt_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["single_stream"]["total"] = cluster_info_dict.copy()

        # Allow override from model_kwargs for ClusCa
        model_kwargs = model_kwargs or {}
        if "clusca_fresh_threshold" in model_kwargs:
            cache_dic["fresh_threshold"] = model_kwargs["clusca_fresh_threshold"]
        if "clusca_cluster_num" in model_kwargs:
            cache_dic["cluster_num"] = model_kwargs["clusca_cluster_num"]
        if "clusca_cluster_method" in model_kwargs:
            cache_dic["cluster_method"] = model_kwargs["clusca_cluster_method"]
        if "clusca_k" in model_kwargs:
            cache_dic["k"] = model_kwargs["clusca_k"]
        if "clusca_propagation_ratio" in model_kwargs:
            cache_dic["propagation_ratio"] = model_kwargs["clusca_propagation_ratio"]

    elif mode == "Hi-ClusCa":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3
        cache_dic["cluster_info"] = {}
        cache_dic["cluster_num"] = 16
        cache_dic["cluster_method"] = "kmeans"
        cache_dic["k"] = 1
        cache_dic["propagation_ratio"] = 0.005
        cache_dic["prediction_mode"] = "hicache"  # Hi-ClusCa uses HiCache predictor

        cluster_info_dict = {}
        cluster_info_dict["cluster_indices"] = None
        cluster_info_dict["centroids"] = None

        cache_dic["cluster_info"]["double_stream"] = {}
        cache_dic["cluster_info"]["single_stream"] = {}
        cache_dic["cluster_info"]["double_stream"]["img_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["double_stream"]["txt_mlp"] = cluster_info_dict.copy()
        cache_dic["cluster_info"]["single_stream"]["total"] = cluster_info_dict.copy()

        # Allow override from model_kwargs for Hi-ClusCa
        model_kwargs = model_kwargs or {}
        if "clusca_fresh_threshold" in model_kwargs:
            cache_dic["fresh_threshold"] = model_kwargs["clusca_fresh_threshold"]
        if "clusca_cluster_num" in model_kwargs:
            cache_dic["cluster_num"] = model_kwargs["clusca_cluster_num"]
        if "clusca_cluster_method" in model_kwargs:
            cache_dic["cluster_method"] = model_kwargs["clusca_cluster_method"]
        if "clusca_k" in model_kwargs:
            cache_dic["k"] = model_kwargs["clusca_k"]
        if "clusca_propagation_ratio" in model_kwargs:
            cache_dic["propagation_ratio"] = model_kwargs["clusca_propagation_ratio"]
        if "hicache_scale" in model_kwargs:
            cache_dic["hicache_scale"] = model_kwargs["hicache_scale"]
        cache_dic["cluster_info"]["single_stream"]["total"] = cluster_info_dict.copy()

    current = {}
    current["final_time"] = timesteps[-2]
    current["activated_steps"] = []

    # Override default values with provided parameters
    if interval is not None:
        cache_dic["fresh_threshold"] = interval
    if max_order is not None:
        cache_dic["max_order"] = max_order
    if first_enhance is not None:
        cache_dic["first_enhance"] = first_enhance
    if hicache_scale is not None:
        cache_dic["hicache_scale_factor"] = hicache_scale

    return cache_dic, current
