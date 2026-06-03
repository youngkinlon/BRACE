from typing import Dict

import math
import torch


def _hicache_polynomial(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    物理学家的 Hermite 多项式 H_n(x)
    使用递推关系: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2 * x

    H_prev = torch.ones_like(x)
    H_curr = 2 * x

    for k in range(2, n + 1):
        H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
        H_prev, H_curr = H_curr, H_next

    return H_curr


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
        # (cache_module.get(i, None) is not None) and
        if  (cache_module.get(i, None) is not None) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_module[i]) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors
def store_feature(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    ache_dic['cache'][-1][current['stream']][current['layer']][current['module']].append(feature)
    if len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]) > cache_dic[capacity]:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].pop(0)


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
    if mode == "hicache":
        return _hicache_prediction_formula(cache_dic, current)
    if mode == "taylor_scaled":
        return _taylor_scaled_prediction_formula(cache_dic, current)
    raise ValueError(f"Unknown prediction_mode: '{mode}'. Must be 'taylor', 'hicache', or 'taylor_scaled'.")


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
    except KeyError as exc:
        raise ValueError(
            f"Cache not found for stream='{current['stream']}', layer={current['layer']}, module='{current['module']}'"
        ) from exc

    # 🔥 修复：使用max_order参数限制使用的项数
    max_order = cache_dic.get("max_order", 3)
    effective_order = min(max_order + 1, len(feats_d))  # +1 because we include 0th order

    for i in range(effective_order):
        output += (1 / math.factorial(i)) * feats_d[i] * (x**i)

    return output


def _hicache_prediction_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    基于 Hermite 多项式的特征预测
    """
    x = current["step"] - current["activated_steps"][-1]

    try:
        feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
    except KeyError as exc:
        raise ValueError(
            f"Cache not found for stream='{current['stream']}', layer={current['layer']}, module='{current['module']}'"
        ) from exc

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
    Taylor 预测的“双重缩放”变体。
    """
    x = current["step"] - current["activated_steps"][-1]

    try:
        feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
    except KeyError as exc:
        raise ValueError(
            f"Cache not found for stream='{current['stream']}', layer={current['layer']}, module='{current['module']}'"
        ) from exc

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

def brace_cache_init(cache_dic: Dict, current: Dict):
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = []
