import json
from typing import Dict
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    layer_cache = cache_dic['cache'][-1][current['stream']][current['layer']]

    # 如果当前模块（如 'img_attn'）还没在字典里，先给它一个空字典
    if current['module'] not in layer_cache:
        layer_cache[current['module']] = {}
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    # 如果当前模块（如 'img_attn'）还没在字典里，先给它一个空字典

    #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature
    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break

    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

GLOBAL_ANALYSIS_LOG = []

import torch
import torch.nn.functional as F
import math
from typing import Dict

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    #x = current['t'] - current['activated_times'][-1]
    output = 0
    UseHicache = False
    if UseHicache:

        # 使用hicache,
        feats_d = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]]
        max_order = cache_dic.get("max_order", 3)
        available_order = len(feats_d) - 1
        order = min(max_order, available_order)
        F_latest = feats_d[0].clone()  # F_0
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
         for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
            output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)

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
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}

def bary_cache_init(cache_dic: Dict, current: Dict):
    if (current['step'] == 0) and (cache_dic['taylor_cache']) :
        # 存储特征；
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = []

def store_feature(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].append(feature)
    if len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])>cache_dic['Capacity'] :
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].pop(0)


def bary_formula_old(cache_dic: Dict, current: Dict) -> torch.Tensor:
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
    w[-1] *= 0.6
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

    return predicted_feature

def bary_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    useFoca=False
    if useFoca:
        return foca_formula(cache_dic, current)
    else:
        return bary_formula_modular(cache_dic, current)

import torch.nn.functional as F
from typing import Dict
import numpy as np
def foca_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    feature_list = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    N = len(feature_list)
    activated_time = current['activated_steps']
    F_pred=4 * feature_list[-1] / 3 -feature_list[-2] / 3 +  2 * (current['step']-activated_time[-1] ) *(feature_list[-1]-feature_list[-2])/((activated_time[-1]-activated_time[-2]) *3)
    F_dri = (3*F_pred-4 * feature_list[-1] + feature_list[-2]) / (2*(current['step']-activated_time[-1]))
    F_pred_correct=feature_list[-1].clone()
    # F_dri_v2=(F_pred-feature_list[-1]) / (current['step']-activated_time[-1])
    # if N==3:
    #     F_pred_correct+=( current['step'] - activated_time[-1] ) * ((feature_list[-1]-feature_list[-2])/(activated_time[-1]-activated_time[-2]) + F_dri) /2
    # else:
    #     F_pred_correct+=((current['step'] - activated_time[-1] )  * F_dri )  / 2
    F_pred_correct+=((current['step'] - activated_time[-1] )  * F_dri ) * 0.386
    return F_pred_correct

def bary_formula_true(cache_dic: Dict, current: Dict) -> torch.Tensor:
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

    # 🛡️ 保护 1: 防止 t_max == t_min (跨度过小)
    if abs(t_max - t_min) < 1e-6:
        return feature_list[-1]

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
            w *= 0.7

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

    for i in range(1, N):
        ci = coeffs[i] / den
        # 等价于 predicted_feature += feature_list[i] * ci
        predicted_feature.add_(feature_list[i], alpha=ci)

    return predicted_feature


def bary_formula_form_i(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    第一类重心插值公式 (Barycentric Form I) - 等价于全局多项式插值
    """
    feature_list = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    N = len(feature_list)

    # 🛡️ 基础保护
    if N <= 1:
        return feature_list[-1]

    activated_time = current['activated_steps']
    steps_for_features = activated_time[-N:]

    # 1. 坐标归一化
    t_target = float(current['step'])
    t_min = float(steps_for_features[0])
    t_max = float(steps_for_features[-1])

    if abs(t_max - t_min) < 1e-6:
        return feature_list[-1]

    # 将 t 映射到 [-1, 1]
    x_target = 2.0 * (t_target - t_min) / (t_max - t_min) - 1.0

    # 2. 计算节点多项式 L(x) = prod(x - x_i)
    # 以及对应的拉格朗日基函数系数
    x_nodes = []
    l_x = 1.0
    for i in range(N):
        t_i = float(steps_for_features[i])
        xi = 2.0 * (t_i - t_min) / (t_max - t_min) - 1.0
        x_nodes.append(xi)

        diff = x_target - xi
        # 🛡️ 奇异点处理：如果目标点重合，直接返回该节点特征
        if abs(diff) < 1e-5:
            return feature_list[i]

        l_x *= diff

    # 3. 计算 Form I 的系数: c_i = L(x) * w_i / (x - x_i)
    # 注意：这里的 w_i 是经典多项式重心权重，对于等距节点 w_i = (-1)^(n-i) * binom(n, i)
    # 为了公平对比，这里采用多项式插值的标准权重计算逻辑
    import math
    def get_poly_weight(i, n):
        # 等距节点下的标准多项式重心权重
        return ((-1) ** (n - 1 - i)) * math.comb(n - 1, i)

    coeffs = []
    for i in range(N):
        # 计算第 i 个节点的重心权重 w_i
        wi = float(get_poly_weight(i, N))

        # Form I 核心公式项
        # 注意：这里不进行分母归一化，直接累加
        ci = l_x * wi / (x_target - x_nodes[i])
        coeffs.append(ci)

    # 4. 特征加权求和
    # 创建初始预测特征
    predicted_feature = feature_list[0] * coeffs[0]

    for i in range(1, N):
        # 这里的 coeffs[i] 可能非常巨大（由于 L(x) 的外推发散）
        predicted_feature.add_(feature_list[i], alpha=coeffs[i])

    return predicted_feature


from typing import Dict, List, Literal
import torch
import math


def bary_formula_modular(
        cache_dic: Dict,
        current: Dict,
        mode: Literal['uniform', 'berrut', 'floater_hormann', 'classic_chebyshev', 'brace_adapted'] = 'floater_hormann',
        gamma: float = 0.7,
        d: int = 1
) -> torch.Tensor:
    # use for ablation study

    feature_list = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    N = len(feature_list)

    if N <= 1:
        return feature_list[-1]

    activated_time = current['activated_steps']
    steps_for_features = activated_time[-N:]

    # 1. 坐标归一化
    t_target = float(current['step'])
    t_min = float(steps_for_features[0])
    t_max = float(steps_for_features[-1])

    if abs(t_max - t_min) < 1e-6:
        return feature_list[-1]

    x_target = 2.0 * (t_target - t_min) / (t_max - t_min) - 1.0

    # 2. 预计算不同模式下的权重 w_j
    weights = []

    if mode == 'uniform':
        # Uniform Weights: w_j = 1 (缺乏交替符号，极易导致分母为0)
        weights = [1.0 for _ in range(N)]

    elif mode == 'berrut':
        # Berrut Weights: w_j = (-1)^j
        weights = [float((-1) ** j) for j in range(N)]

    elif mode == 'floater_hormann':
        # Floater-Hormann Weights
        for j in range(N):
            w_j = 0.0
            for k in range(max(0, j - d), min(j, N - 1 - d) + 1):
                temp = math.comb(d, j - k)
                if k % 2 == 1:
                    w_j -= temp
                else:
                    w_j += temp
            weights.append(float(w_j))

    elif mode == 'classic_chebyshev':
        # Classic Chebyshev (Gamma = 0.5)
        for i in range(N):
            w = -1.0 if (i + 1) % 2 != 0 else 1.0
            if i == 0 or i == N - 1:
                w *= 0.5
            weights.append(w)

    elif mode == 'brace_adapted':
        # BRACE Adapted (可调 Gamma)
        for i in range(N):
            w = -1.0 if (i + 1) % 2 != 0 else 1.0
            if i == 0:
                w *= 0.5
            if i == N - 1:
                w *= gamma
            weights.append(w)

    # 3. 计算重心系数
    coeffs = []
    den = 0.0
    for i in range(N):
        t_i = float(steps_for_features[i])
        x_i = 2.0 * (t_i - t_min) / (t_max - t_min) - 1.0
        diff = x_target - x_i

        # 奇异点保护
        if abs(diff) < 1e-5:
            return feature_list[i]

        # 权重计算
        temp = weights[i] / diff
        coeffs.append(temp)
        den += temp

    # 🛡️ 保护：防止分母为 0
    if abs(den) < 1e-8:
        den = 1e-8

    # 4. 显存高效的加权求和
    c0 = coeffs[0] / den
    predicted_feature = feature_list[0] * c0

    for i in range(1, N):
        ci = coeffs[i] / den
        predicted_feature.add_(feature_list[i], alpha=ci)

    return predicted_feature
