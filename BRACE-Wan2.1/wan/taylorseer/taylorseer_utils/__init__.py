from typing import Dict 
import torch
import math

def store_feature(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].append(feature)

    if len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]) > cache_dic['capacity']:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].pop(0)

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
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

def taylor_formula(derivative_dict: Dict, distance: int) -> torch.Tensor:
    """
    Compute Taylor expansion error.
    
    :param derivative_dict: Derivative dictionary
    :param x: Current step
    """
    output=0
    for i in range(len(derivative_dict)):
        output += (1 / math.factorial(i)) * derivative_dict[i] * (distance ** i)
    return output

def brace_formula(derivative_list: list, distance: int,current) -> torch.Tensor:
    useFoca = False
    if useFoca:
        return foca_formula(derivative_list, distance,current=current)
    else:
        return bary_formula(derivative_list, distance,current=current)

def foca_formula(derivative_list: list, distance: int,current) -> torch.Tensor:
    feats_list = derivative_list
    if len(feats_list) < 2:
        return feats_list[0]  # 不足两帧，直接返回当前值

    # 获取历史值：F_n 和 F_{n-1}
    F_n = feats_list[-1]
    F_nm1 = feats_list[-2]
    # Step 1: BDF2 预测
    F_pred = (4.0 * F_n - F_nm1) / 3.0

    # Step 2: Heun 修正
    F_out = 0.5 * (F_n + F_pred)

    return F_out

def bary_formula(derivative_list: list, distance: int,current) -> torch.Tensor:
    feature_list = derivative_list
    N = len(feature_list)
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

        if abs(diff) < 1e-5:
            return feature_list[i]

        diff += 1e-8 if diff >= 0 else -1e-8


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

    c0 = coeffs[0] / den
    predicted_feature = feature_list[0] * c0

    for i in range(1, N):
        ci = coeffs[i] / den
        predicted_feature.add_(feature_list[i], alpha=ci)
    return predicted_feature

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """

    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


def brace_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = []

