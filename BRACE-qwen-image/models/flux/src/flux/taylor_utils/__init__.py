from typing import Dict
import torch
import math


def _hicache_polynomial(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    物理学家的 Hermite 多项式 H_n(x)
    使用递推关系: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    """
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2 * x

    H_prev = torch.ones_like(x)
    H_curr = 2 * x

    for k in range(2, n + 1):
        H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
        H_prev, H_curr = H_curr, H_next

    return H_curr


def _update_analytic_stats(cache_dic: Dict, layer_key: int, delta_tensor: torch.Tensor):
    """
    Update analytic statistics for adaptive sigma calculation.
    """
    if not cache_dic.get("analytic_sigma", False):
        return

    # Get config
    config = cache_dic.get("analytic_sigma_config", {})
    beta = config.get("beta", 0.01)

    # beta <= 0: 视为“关闭在线更新”，保持解析 σ 使用默认 q=1
    # 即不写入任何 analytic_stats，让 _compute_analytic_sigma 走默认分支。
    if beta <= 0:
        return

    # Calculate scalar scale D
    # delta_tensor shape: [Batch, Sequence, Dim] or [Sequence, Dim]
    # We want a robust measure of magnitude.
    with torch.no_grad():
        norms = delta_tensor.detach().float().norm(dim=-1)
        q_quantile = config.get("q_quantile")
        if q_quantile is not None and 0.0 < float(q_quantile) < 1.0:
            # 使用分位数以抵抗少量 outlier
            D_val = torch.quantile(norms, float(q_quantile)).item()
        else:
            D_val = norms.mean().item()

    # Initialize storage if needed
    if "analytic_stats" not in cache_dic:
        cache_dic["analytic_stats"] = {}

    stats = cache_dic["analytic_stats"]

    if layer_key not in stats:
        # Initialize
        stats[layer_key] = D_val
    else:
        # EMA update: q = sqrt( (1-beta)*q^2 + beta*D^2 )
        q_old = stats[layer_key]
        # Use squared average EMA for RMS-like behavior
        q_sq_old = q_old**2
        q_sq_new = (1 - beta) * q_sq_old + beta * (D_val**2)
        stats[layer_key] = math.sqrt(q_sq_new)


def _compute_analytic_sigma(cache_dic: Dict, layer_key: int) -> float:
    """
    Compute analytic sigma based on collected statistics.
    """
    stats = cache_dic.get("analytic_stats", {})
    # If no stats collected yet (e.g. first step), fallback to a default scale
    # Or use a safe default like 1.0 for q, which implies sigma ~ alpha/1.414
    q = stats.get(layer_key, 1.0)

    config = cache_dic.get("analytic_sigma_config", {})
    alpha = config.get("alpha", 1.28)
    sigma_max = config.get("sigma_max", 1.0)
    eps = config.get("eps", 1e-6)
    sigma_smooth = config.get("sigma_smooth", 0.0)

    # Formula: sigma = min(sigma_max, alpha / (sqrt(2)*q + eps))
    raw = alpha / (math.sqrt(2) * q + eps)
    sigma_new = min(sigma_max, raw)

    # 可选：在 log 域对 sigma 做平滑，避免相邻步跳变
    if sigma_smooth and sigma_smooth > 0:
        state = cache_dic.setdefault("analytic_sigma_state", {})
        prev = state.get(layer_key, sigma_new)
        gamma = float(sigma_smooth)
        log_prev = math.log(max(prev, eps))
        log_new = math.log(max(sigma_new, eps))
        smoothed = math.exp((1 - gamma) * log_prev + gamma * log_new)
        sigma_new = min(sigma_max, smoothed)
        state[layer_key] = sigma_new

    return sigma_new



def _collect_trajectory_feature(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    特征轨迹收集器 - 在缓存更新时自动收集特征
    支持多层、多模块同时收集（一次推理收集所有需要的模块）

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Feature tensor to collect
    """
    config = cache_dic.get("feature_collection_config", {})

    # 支持多层特征收集
    target_layers = config.get("target_layers", [14])
    if isinstance(target_layers, int):
        target_layers = [target_layers]  # 向后兼容

    # 检查收集条件 - 只有在目标层列表中才收集
    if current["layer"] not in target_layers:
        return

    # 🔥 新增：支持多模块同时收集
    target_modules = config.get("target_modules", ["any"])
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    # 检查模块是否需要收集
    if "any" not in target_modules and current["module"] not in target_modules:
        return

    # 支持多流收集
    target_streams = config.get("target_streams", ["any"])
    if isinstance(target_streams, str):
        target_streams = [target_streams]

    # 检查流是否需要收集
    if "any" not in target_streams and current["stream"] not in target_streams:
        return

    # 🔥 新增：按层-模块组合存储，支持同时收集多个模块
    if "trajectory_features" not in cache_dic:
        cache_dic["trajectory_features"] = {}
        cache_dic["trajectory_metadata"] = {}

    layer_key = current["layer"]
    module_key = current["module"]

    # 初始化层级存储
    if layer_key not in cache_dic["trajectory_features"]:
        cache_dic["trajectory_features"][layer_key] = {}
        cache_dic["trajectory_metadata"][layer_key] = {}

    # 初始化模块级存储
    if module_key not in cache_dic["trajectory_features"][layer_key]:
        cache_dic["trajectory_features"][layer_key][module_key] = []
        cache_dic["trajectory_metadata"][layer_key][module_key] = []

    # 收集特征到对应的模块存储中
    cache_dic["trajectory_features"][layer_key][module_key].append(feature.clone().detach().cpu())
    cache_dic["trajectory_metadata"][layer_key][module_key].append(
        {
            "step": current["step"],
            "timestep": current.get("t", 0),
            "cache_type": current.get("type", "full"),
            "layer": current["layer"],
            "module": current["module"],
            "stream": current["stream"],
        }
    )

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    # 🔥 新增：特征收集钩子
    if cache_dic.get("enable_feature_collection", False):
        _collect_trajectory_feature(cache_dic, current, feature)
        # 🔥 如果只是为了特征收集，跳过其余的缓存操作
        if not cache_dic.get("taylor_cache", False):
            return

    # 🔥 安全检查：确保缓存结构存在
    try:
        cache_module = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
    except KeyError:
        return

    # 🔥 安全检查：确保有足够的 activated_steps 用于计算差分
    if len(current["activated_steps"]) < 2:
        # 即便差分不够，也要记录零阶特征，供下一次缓存预测使用
        cache_module[0] = feature
        return

    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]
    # difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_module.get(i, None) is not None) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_module[i]
            ) / difference_distance
            
            # 🔥 Analytic Sigma Update: Capture first-order difference (delta_F)
            if i == 0:
                _update_analytic_stats(cache_dic, current["layer"], updated_taylor_factors[1])
                
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def get_collected_features(cache_dic: Dict) -> tuple:
    """
    获取收集的特征轨迹 - 支持多层数据

    :param cache_dic: Cache dictionary
    :return: (features_dict, metadata_dict) tuple where keys are layer indices
    """
    features = cache_dic.get("trajectory_features", {})
    metadata = cache_dic.get("trajectory_metadata", {})
    return features, metadata


def clear_collected_features(cache_dic: Dict):
    """
    清空收集的特征轨迹

    :param cache_dic: Cache dictionary
    """
    if "trajectory_features" in cache_dic:
        del cache_dic["trajectory_features"]
    if "trajectory_metadata" in cache_dic:
        del cache_dic["trajectory_metadata"]


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Feature prediction dispatcher: chooses between Taylor or HiCache prediction.

    :param cache_dic: Cache dictionary
        - 'prediction_mode': 'taylor' or 'hicache'. Defaults to 'taylor' if not specified.
        - 'use_hicache': (Legacy) If True and 'prediction_mode' is not set, mode becomes 'hicache'.
    :param current: Information of the current step
    """
    # Determine prediction mode, with backward compatibility for 'use_hicache'
    if "prediction_mode" in cache_dic:
        mode = cache_dic["prediction_mode"]
    elif cache_dic.get("use_hicache", False):
        mode = "hicache"
    else:
        mode = "taylor"

    # Dispatch based on mode
    if mode == "taylor":
        return _taylor_expansion_formula(cache_dic, current)

    elif mode == "hicache":
        return _hicache_prediction_formula(cache_dic, current)

    elif mode == "taylor_scaled":
        return _taylor_scaled_prediction_formula(cache_dic, current)

    else:
        raise ValueError(
            f"Unknown prediction_mode: '{mode}'. Must be 'taylor', 'hicache', or 'taylor_scaled'."
        )


def _taylor_expansion_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    标准泰勒展开预测
    使用幂函数基: F_pred = F_0 + Σ (1/k!) * x^k * Δ^kF
    """
    x = current["step"] - current["activated_steps"][-1]
    # x = current['t'] - current['activated_times'][-1]
    output = 0

    # 🔥 修复：安全检查，确保缓存结构存在
    try:
        feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
    except KeyError:
        # 🔥 修复：如果缓存不存在，说明这是第一次访问该模块
        # 在这种情况下，我们应该初始化缓存结构并返回零张量
        if current["stream"] not in cache_dic["cache"][-1]:
            cache_dic["cache"][-1][current["stream"]] = {}
        if current["layer"] not in cache_dic["cache"][-1][current["stream"]]:
            cache_dic["cache"][-1][current["stream"]][current["layer"]] = {}
        if current["module"] not in cache_dic["cache"][-1][current["stream"]][current["layer"]]:
            cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = {}

        # 🔥 修复：如果缓存为空，这意味着这是第一次调用该模块
        # 在这种情况下，我们应该抛出一个更友好的错误信息，指导用户检查配置
        raise ValueError(
            f"Cache not found for stream='{current['stream']}', layer={current['layer']}, module='{current['module']}'. "
            f"This usually means the first step was not run in 'full' mode to initialize the cache. "
            f"Please check your cache configuration and ensure first_enhance >= 1."
        )

    # 🔥 修复：使用max_order参数限制使用的项数
    max_order = cache_dic.get("max_order", 3)
    effective_order = min(max_order + 1, len(feats_d))  # +1 because we include 0th order

    for i in range(effective_order):
        output += (1 / math.factorial(i)) * feats_d[i] * (x**i)

    return output


def _hicache_prediction_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    基于 Hermite 多项式的特征预测

    使用 Hermite 多项式作为基函数，而不是标准的幂函数：
    F_pred = F_0 + Σ_{k=1}^{n} (1/k!) * H_k(x) * Δ^kF

    其中：
    - H_k(x) 是 k 阶 Hermite 多项式
    - Δ^kF 是 k 阶差分特征
    - x 是时间步差值

    这种方法的优势：
    1. Hermite 多项式具有正交性，数值稳定性更好
    2. 在某些函数类型上逼近精度更高
    3. 通过缩放因子可以控制多项式增长
    """
    # 获取实际的时间步差值，保持与原始泰勒展开的一致性
    x = current["step"] - current["activated_steps"][-1]
    # 如果需要使用时间差值，可以取消注释下面这行
    # x = current['t'] - current['activated_times'][-1]

    # 获取特征缓存
    try:
        feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
    except KeyError:
        raise ValueError(
            f"Cache not found for stream='{current['stream']}', layer={current['layer']}, module='{current['module']}'"
        )

    # 🔥 修复：使用max_order参数限制阶数
    max_order = cache_dic.get("max_order", 3)
    available_order = len(feats_d) - 1  # 可用阶数 = 历史项数 - 1
    order = min(max_order, available_order)  # 使用较小值

    if order < 1:
        return feats_d.get(0)  # 历史不足，返回最新特征

    F_latest = feats_d[0].clone()  # F_0

    # 将时间步差值转换为tensor，保持与特征相同的dtype和device
    x_tensor = torch.tensor(float(x), dtype=F_latest.dtype, device=F_latest.device)

    # 获取缩放因子，用于控制 Hermite 多项式的增长
    if cache_dic.get("analytic_sigma", False):
        scale_factor = _compute_analytic_sigma(cache_dic, current["layer"])
    else:
        scale_factor = cache_dic.get("hicache_scale_factor", 0.5)
        
    x_scaled = x_tensor * scale_factor

    # 构造 Hermite 预测
    pred = F_latest.clone()

    for k in range(1, order + 1):
        diff_k = feats_d[k]
        Hk = _hicache_polynomial(x_scaled, k)
        # 考虑缩放因子的影响
        alpha = float(Hk / math.factorial(k)) * (scale_factor**k)
        pred.add_(diff_k, alpha=alpha)

    return pred


def _taylor_scaled_prediction_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Taylor 预测的“双重缩放”变体：

    在标准幂函数基的基础上进行双重缩放：
      F_pred = F_0 + Σ_{k=1..n} (1/k!) * (s x)^k * (s^k) * Δ^kF
             = F_0 + Σ_{k=1..n} (1/k!) * (s^(2k)) * (x^k) * Δ^kF

    其中 s = hicache_scale_factor, x 为步距（当前步与最近一次 full 步的差）。
    该形式用于对比 Hermite 双重缩放与普通多项式基的效果差异。
    """
    x = current["step"] - current["activated_steps"][-1]

    try:
        feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
    except KeyError:
        raise ValueError(
            f"Cache not found for stream='{current['stream']}', layer={current['layer']}, module='{current['module']}'"
        )

    max_order = cache_dic.get("max_order", 3)
    available_order = len(feats_d) - 1
    order = min(max_order, available_order)
    if order < 1:
        return feats_d.get(0)

    F_latest = feats_d[0].clone()
    x_tensor = torch.tensor(float(x), dtype=F_latest.dtype, device=F_latest.device)
    scale = cache_dic.get("hicache_scale_factor", 0.5)

    pred = F_latest.clone()
    for k in range(1, order + 1):
        diff_k = feats_d[k]
        # 双重缩放：系数 = (1/k!) * (s^(2k)) * (x^k)
        alpha = (float(x_tensor**k) / math.factorial(k)) * (scale ** (2 * k))
        pred.add_(diff_k, alpha=alpha)

    return pred

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = {}
