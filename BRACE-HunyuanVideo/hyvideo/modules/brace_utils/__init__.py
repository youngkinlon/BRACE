from typing import Dict
import torch
import math
from .convert_flops import convert_flops
import numpy as np
import json
try:
    with open("python_gamma_lut_n6.json", "r", encoding='utf-8') as f:
        GLOBAL_GAMMA_LUT = json.load(f)
    print("✅ 成功加载多维 Gamma LUT 2表")
except FileNotFoundError:
    GLOBAL_GAMMA_LUT = {}
    print("⚠️ 未找到 python_gamma_lut.json，将使用默认 fallback 参数")
try:
    with open("python_gamma_lut_stream_step_n6.json", "r", encoding='utf-8') as f:
        GLOBAL_GAMMA_LUT1 = json.load(f)
    print("✅ 成功加载多维 Gamma LUT 2表")
except FileNotFoundError:
    GLOBAL_GAMMA_LUT1 = {}
    print("⚠️ 未找到 python_gamma_lut.json，将使用默认 fallback 参数")
# python_gamma_step_layer_stats_n6.json step , save to 2 round
try:
    with open("python_gamma_step_mean_n6.json", "r", encoding='utf-8') as f:
        GLOBAL_GAMMA_LUT2 = json.load(f)
    print("✅ 成功加载多维 Gamma LUT 2表")
except FileNotFoundError:
    GLOBAL_GAMMA_LUT2 = {}
    print("⚠️ 未找到 python_gamma_lut.json，将使用默认 fallback 参数")
def get_lut_gamma_step(step, default_gamma=1.6):
    if not GLOBAL_GAMMA_LUT2:
        return default_gamma
    T =  str(step)
    try:
        node_data = GLOBAL_GAMMA_LUT2[T]
        return node_data["gamma"]
    except KeyError:
        # 如果某些 layer/step 组合在 profiling 时没有抓到，使用默认值
        return default_gamma

def get_lut_gamma2(layer, module, stream, step, default_gamma=1.6):
    """
    O(1) 查表函数，带有高方差保护机制
    """
    if not GLOBAL_GAMMA_LUT:
        return default_gamma

    L, M, S, T = str(layer), str(module), str(stream), str(step)

    try:
        node_data = GLOBAL_GAMMA_LUT[L][M][S][T]
        return node_data["gamma"]
    except KeyError:
        # 如果某些 layer/step 组合在 profiling 时没有抓到，使用默认值
        return default_gamma
def get_lut_gamma1(stream, step, default_gamma=1.6):
    """
    O(1) 查表函数 (基于 Stream 和 Step)

    参数:
        stream (str): 特征流的类型，例如 'txt' 或 'img', 'single', 'double' 等，需与 LUT 中的键匹配。
        step (int/str): 当前的降噪步数。
        default_gamma (float): 如果查不到数据时的默认回退值。
                               建议使用 1.0 (无额外缩放)，而不是之前的 1.6。

    返回:
        float: 查表得到的最优 gamma 值。
    """
    if not GLOBAL_GAMMA_LUT1:
        return default_gamma

    # 强制转换为字符串，确保能匹配 JSON 里的键格式
    S, T = str(stream), str(int(step))

    try:
        node_data = GLOBAL_GAMMA_LUT1[S][T]

        return node_data["gamma"]

    except KeyError:
        # 遇到没有 profiling 到的特殊 stream 或 step 时，安全回退
        return default_gamma

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


import torch.nn.functional as F
def find_best_arg(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    x = current['step'] - current['activated_steps'][-1]
    cached_feats = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
    max_cache_len = min(len(cached_feats), 2)
    gammas = [round(0.1 + i * 0.2, 2) for i in range(30)]
    best_gamma = 1.0
    min_direction_error = float('inf')  # 现在的目标仅仅是方向误差
    best_output = None
    target_feature = feature.float()
    for gamma in gammas:
        alphas = []
        for k in range(max_cache_len):
            if k == 0:
                alphas.append(1.0)
            else:
                alpha_k = 1.0 / math.gamma(gamma * k + 1.0)
                alphas.append(alpha_k)
        # 组装预测特征
        output = (alphas[0] * cached_feats[0]).clone()
        for i in range(1, max_cache_len):
            output += alphas[i] * cached_feats[i].clone() * (x ** i)
        output_f32 = output
        cos_sim = F.cosine_similarity(output_f32, target_feature, dim=-1).mean()
        direction_error = (1.0 - cos_sim).item()

        # 现在的决策完全不参考 magnitude_error
        if direction_error < min_direction_error:
            min_direction_error = direction_error
            best_gamma = gamma
            best_output = output_f32

    return best_gamma, min_direction_error, best_output

GLOBAL_ANALYSIS_LOG = []

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    output = 0
    UseHicache = False
    UseFora = False
    UseFade = True
    if UseHicache:
        feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
        max_order = cache_dic.get("max_order", 3)
        available_order = len(feats_d) - 1
        order = min(max_order, available_order)
        F_latest = feats_d[0].clone()
        x_tensor = torch.tensor(float(x), dtype=F_latest.dtype, device=F_latest.device)
        scale_factor = cache_dic.get("hicache_scale_factor", 0.5)

        x_scaled = x_tensor * scale_factor
        pred = F_latest.clone()
        for k in range(1, order + 1):
            diff_k = feats_d[k]
            Hk = _hicache_polynomial(x_scaled, k)
            # 考虑缩放因子的影响
            alpha = float(Hk / math.factorial(k)) * (scale_factor ** k)
            pred.add_(diff_k, alpha=alpha)
        output=pred
    else:
        if UseFora:
            output += cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0]
        elif UseFade:
            feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
            max_cache_len = len(feats_d)
            best_gamma = get_lut_gamma2(current['layer'], current['module'], current['stream'], current['step'],
                                        default_gamma=1.6)
            output = feats_d[0].clone()
            x_val = float(x)
            best_gamma1 = get_lut_gamma1(current['stream'], current['step'],
                                        default_gamma=1.6)
            best_gamma_step = get_lut_gamma_step(current['step'])
            for k in range(1, max_cache_len):
                denom = math.gamma(best_gamma_step * k + 1.0)
                # 最终的标量 alpha = (x^k) / Γ(γ*k + 1)
                alpha_scalar = ((x_val) ** k) / denom
                output.add_(feats_d[k], alpha=float(alpha_scalar))
        else:
            for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
                output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
    #fora
    #output+=cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0]
    return output

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

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache, expanding storage areas for Taylor series derivatives
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current['step'] == 0:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}

def bary_cache_init(cache_dic: Dict, current: Dict):
    if current['step'] == 0:
        # 存储特征；
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = []

def store_feature(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].append(feature)
    if len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])>cache_dic['c'] :
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].pop(0)


def bary_formula_fast(cache_dic: Dict, current: Dict) -> torch.Tensor:
    feature_list = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    activated_time = current['activated_steps']
    steps_for_features = activated_time[-len(feature_list):]
    features = torch.stack(feature_list)

    device = features.device
    dtype = features.dtype

    t = torch.tensor(steps_for_features, device=device, dtype=dtype)
    t_target = torch.tensor(current['step'], device=device, dtype=dtype)

    t_min, t_max = t.min(), t.max()

    # 🛡️ 保护 1: 防止 t_max == t_min (只有一个点时)
    if (t_max - t_min).abs() < 1e-6:
        return feature_list[-1]  # 如果时间跨度太小，无法插值，直接返回最近的

    x = 2.0 * (t - t_min) / (t_max - t_min) - 1.0
    x_target = 2.0 * (t_target - t_min) / (t_max - t_min) - 1.0

    # Chebyshev 权重
    j = torch.arange(len(feature_list), dtype=dtype, device=device)
    w = (-1) ** (j + 1)
    w[0] *= 0.5
    w[-1] *= 0.4

    diff = x_target - x

    # 🛡️ 保护 2: 处理奇异点 (当 x_target 恰好等于某个 x 时)
    # 如果 diff 非常小，说明我们正在预测一个已知点，直接返回该点特征
    mask_singularity = diff.abs() < 1e-5
    if mask_singularity.any():
        # 找到最近的那个索引
        idx = mask_singularity.nonzero(as_tuple=True)[0][0]
        return feature_list[idx]

    # 🛡️ 保护 3: 防止除以零产生的数值爆炸
    # 给 diff 加一个极小的 epsilon，虽然上面的 check 应该已经拦住了
    diff = diff + 1e-8 * diff.sign()

    temp = w / diff
    den = torch.sum(temp)

    # 🛡️ 保护 4: 防止分母为 0
    if den.abs() < 1e-6:
        den = 1e-6

    view_shape = [len(feature_list)] + [1] * (features.dim() - 1)
    temp_expanded = temp.view(*view_shape)

    num = torch.sum(temp_expanded * features, dim=0)

    predicted_feature = num / den

    # 🛡️ 保护 5: 最终检查 NaN (调试用，稳定后可删除)
    if torch.isnan(predicted_feature).any() or torch.isinf(predicted_feature).any():
        print(f"⚠️ NaN detected at Step {current['step']}! Fallback to last feature.")
        return feature_list[-1]

    return predicted_feature

def bary_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    feature_list = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    N = len(feature_list)

    # 🛡️ 如果特征数量太少，直接返回
    if N <= 1:
        return feature_list[-1]

    activated_time = current['activated_steps']
    steps_for_features = activated_time[-N:]

    # 1. 全部转换为纯 Python float 进行标量运算，完全避开 PyTorch 框架开销
    t_target = float(current['step'])
    t_min = float(steps_for_features[0])
    t_max = float(steps_for_features[-1])

    x_target = 2.0 * (t_target - t_min) / (t_max - t_min) - 1.0

    coeffs = []
    den = 0.0

    # 2. 显性 for 循环计算系数
    for i in range(N):
        t_i = float(steps_for_features[i])
        x_i = 2.0 * (t_i - t_min) / (t_max - t_min) - 1.0

        diff = x_target - x_i

        # 🛡️ 保护 2: 处理奇异点
        if abs(diff) < 1e-5:
            return feature_list[i]

        # 🛡️ 保护 3: 防止除以零
        diff += 1e-8 if diff >= 0 else -1e-8

        # Chebyshev 权重逻辑
        w = -1.0 if (i + 1) % 2 != 0 else 1.0  # i=0 -> -1; i=1 -> 1; i=2 -> -1
        if i == 0:
            w *= 0.5
        if i == N - 1:
            w *= 0.6

        temp = w / diff
        coeffs.append(temp)
        den += temp

    # 🛡️ 保护 4: 防止分母为 0
    if abs(den) < 1e-6:
        den = 1e-6

    # 3. 高效的特征组合 (避免 torch.stack)
    # 创建初始预测特征 (这会分配一次新显存)
    c0 = coeffs[0] / den
    predicted_feature = feature_list[0] * c0
    # 使用原地操作 (in-place) 加上剩余的特征，极其节省显存和带宽
    for i in range(1, N):
        ci = coeffs[i] / den
        # 等价于 predicted_feature += feature_list[i] * ci
        predicted_feature.add_(feature_list[i], alpha=ci)

    return predicted_feature