import json
import os
import pickle
import random
from itertools import product
from operator import itemgetter
from os import times
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import math
import time
import torch.nn.functional as F
from sympy.physics.units import temperature



from collections import defaultdict
GLOBAL_ANALYSIS_LOG = []
def store_feature(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    cache_dic['cache'][current['layer']][current['module']].append(feature)
    if len(cache_dic['cache'][current['layer']][current['module']]) > 4:
        cache_dic['cache'][current['layer']][current['module']].pop(0)

def fora_formula(cache_dic: Dict, current: Dict):
    return cache_dic['cache'][current['layer']][current['module']][-1]


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    """计算导数近似值，存储到缓存中"""
    # 计算两次全量计算步骤的间隔（步长差）
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    # 存储各阶导数的字典，0阶导数即当前全量计算的特征值
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature # 0阶导数 = 当前特征值


    #计算1阶及更高阶导数（最高阶数由cache_dic['max_order']限制）
    for i in range(cache_dic['max_order']):
        # 检查是否有上一次的i阶导数缓存，且当前步骤不在后期增强阶段
        if (cache_dic['cache'][-1][current['layer']][current['module']].get(i, None) is not None) and (current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)):
            # 用差分近似导数：(当前i阶导数 - 上一次i阶导数) / 步长差
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    #将计算得到的各阶导数更新到缓存中
    cache_dic['cache'][-1][current['layer']][current['module']] = updated_taylor_factors

import torch.nn.functional as F

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    # global cumulative_time, call_count
    # start = time.perf_counter()
    x = current['step'] - current['activated_steps'][-1]
    # x = current['t'] - current['activated_times'][-1]
    output = 0
    UseHicache=False
    UseNewIdea= True
    UseADI= False
    UseNewTraining = False
    if UseHicache:
        feats_d=cache_dic['cache'][-1][current['layer']][current['module']]
        max_order = cache_dic.get("max_order", 4)
        available_order = len(feats_d) - 1  # 可用阶数 = 历史项数 - 1
        order = min(max_order, available_order)  # 使用较小值
        F_latest = feats_d[0].clone()  # F_0
        x_tensor = torch.tensor(float(x), dtype=F_latest.dtype, device=F_latest.device)
        scale_factor = cache_dic.get("hicache_scale_factor", 0.75)
        x_scaled = x_tensor * scale_factor
        pred = F_latest.clone()
        for k in range(1, order + 1):
            diff_k = feats_d[k]
            Hk = _hicache_polynomial(x_scaled, k)
            # 考虑缩放因子的影响
            alpha = float(Hk / math.factorial(k)) * (scale_factor ** k)
            pred.add_(diff_k, alpha=alpha)
        output = pred
    else:
        if UseNewIdea:
            feats_d = cache_dic['cache'][-1][current['layer']][current['module']]
            max_order = cache_dic.get("max_order", 4)
            available_order = len(feats_d) - 1  # 可用阶数 = 历史项数 - 1
            order = min(max_order, available_order)  # 使用较小值

            # 1. 提取 0 阶基底特征（绝对安全的起点）
            F_latest = feats_d[0].clone()  # F_0
            pred = F_latest.clone()
            for k in range(1, order + 1):
                diff_k = feats_d[k]
                gamma = 1.0
                s = math.copysign(abs(x) ** gamma, x) if x != 0 else 0.0
                factor = abs(current['real_timestep']-current['activated_steps_real']) / (20.0*abs(x))
                alpha_taylor = float(1.0 / math.factorial(k)) * ( (s* factor) ** k )
                alpha_laurent = 1.0 / (abs(s * factor) + 1 ) ** k
                alpha = alpha_taylor - alpha_laurent
                # 将第 k 阶特征按权重累加到预测张量中
                pred.add_(diff_k, alpha=alpha)
            output = pred


        else:
            for i in range(len(cache_dic['cache'][-1][current['layer']][current['module']])):
            # 泰勒展开项：(1/i!) * i阶导数 * x^i
                output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['layer']][current['module']][i] * (x ** i)
    return output

def new_idea(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    x = current['step'] - current['activated_steps'][-1]
    output = 0
    for i in range(len(cache_dic['cache'][-1][current['layer']][current['module']])):
        # 泰勒展开项：(1/i!) * i阶导数 * x^i
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['layer']][current['module']][i] * (x ** i)
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

def berk_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    feature_list = cache_dic['cache'][current['layer']][current['module']]
    activated_time = current['activated_steps']
    k = len(feature_list)
    steps_for_features = activated_time[-k:]
    features = torch.stack(feature_list)
    # 2. 设备与数据类型对齐
    device = features.device
    dtype = features.dtype
    # T：实际时间步张量
    t = torch.tensor(steps_for_features, device=device, dtype=dtype)
    t_target = torch.tensor(current['step'], device=device, dtype=dtype)
    t_min, t_max = t.min(), t.max()  # t_min=最小值, t_max=最大值（49）
    x = 2.0 * (t - t_min) / (t_max - t_min) - 1.0
    x_target = 2.0 * (t_target - t_min) / (t_max - t_min) - 1.0

    # 4. Chebyshev 闭式权重（最快、最稳）
    j = torch.arange(k, dtype=dtype, device=device)
    w = (-1) ** (j+1)
    w[0] *= 0.5  # 端点权重减半
    w[-1] *= 0.5
    diff = x_target - x
    temp = w / diff  # 形状: [k]
    # d. 计算分母 den (Denominator): 所有 temp 求和
    den = torch.sum(temp)  # 标量
    view_shape = [k] + [1] * (features.dim() - 1)
    temp_expanded = temp.view(*view_shape)
    # 分子求和： temp_expanded * features
    num = torch.sum(temp_expanded * features, dim=0)  # 形状: [D1, D2, D3, ...]
    # 结果计算 (标量 den 与张量 num 的广播除法)
    predicted_feature = num / den
    return predicted_feature


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and expand storage for different-order derivatives.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    if current['step'] == (current['num_steps'] - 1):
        cache_dic['cache'][-1][current['layer']][current['module']] = {}
        cache_dic['cache'][-1][current['layer']][current['module']]['pred1']=None
        cache_dic['cache'][-1][current['layer']][current['module']]['pred2']=None
        cache_dic['cache'][-1][current['layer']][current['module']]['res'] = None
        cache_dic['cache'][current['layer']][current['module']]=[]



def bary_formula_fast(cache_dic: Dict, current: Dict) -> torch.Tensor:
    feature_list = cache_dic['cache'][current['layer']][current['module']]
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
            w *= 0.5

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
