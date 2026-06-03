from typing import Dict 
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current['mode']=='Brace':
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].append(feature)
        if len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]) > cache_dic['capacity']:
            cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].pop(0)
        return
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


def brace_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
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


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current['mode']=='Brace':
        return brace_formula(cache_dic=cache_dic, current=current)

    x = current['step'] - current['activated_steps'][-1]
    #x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
    
    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current['mode'] == 'Brace':
        if (current['step'] == 0) and (cache_dic['taylor_cache']):
            cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = []

    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}
