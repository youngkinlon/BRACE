import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules import WanModel
from wan.modules.model import sinusoidal_embedding_1d, WanAttentionBlock
from wan.modules.attention import flash_attention

from wan.taylorseer.taylorseer_utils import taylor_cache_init, derivative_approximation, taylor_formula, \
    brace_cache_init, store_feature
from .wan_attention_cache_forward import wan_attention_cache_forward,wan_attention_cache_forward_brace

def wan_attention_forward(
    self:WanAttentionBlock,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
    cache_dic,
    current
):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, 6, C]
        seq_lens(Tensor): Shape [B], length of each sequence in batch
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    assert e.dtype == torch.float32
    with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
    assert e[0].dtype == torch.float32

    if current['type'] == 'full':
        
        # self-attention
        current['module'] = 'self-attention'
        if current['mode'] == 'Taylor':
            taylor_cache_init(cache_dic=cache_dic, current=current)
        elif current['mode'] == 'Brace':
            brace_cache_init(cache_dic=cache_dic, current=current)
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        if current['mode'] == 'Taylor':
            derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        elif current['mode'] == 'Brace':
            store_feature(cache_dic=cache_dic, current=current, feature=y)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention 
        current['module'] = 'cross-attention'
        if current['mode'] == 'Taylor':
            taylor_cache_init(cache_dic=cache_dic, current=current)
        elif current['mode'] == 'Brace':
            brace_cache_init(cache_dic=cache_dic, current=current)
        y = self.cross_attn(self.norm3(x), context, context_lens)
        #derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        if current['mode'] == 'Taylor':
            derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        elif current['mode'] == 'Brace':
            store_feature(cache_dic=cache_dic, current=current, feature=y)
        x = x + y

        # ffn
        current['module'] = 'ffn'
        #taylor_cache_init(cache_dic=cache_dic, current=current)
        if current['mode'] == 'Taylor':
            taylor_cache_init(cache_dic=cache_dic, current=current)
        elif current['mode'] == 'Brace':
            brace_cache_init(cache_dic=cache_dic, current=current)

        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        #derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        if current['mode'] == 'Taylor':
            derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        elif current['mode'] == 'Brace':
            store_feature(cache_dic=cache_dic, current=current, feature=y)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

    elif current['type'] == 'Taylor':
        
        #x = wan_attention_cache_forward(cache_dic, current, e, x)
        if current['mode']== 'Taylor':
            x = wan_attention_cache_forward(
                sa_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['self-attention'],
                ca_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['cross-attention'],
                ffn_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['ffn'],
                e=e,
                x=x,
                distance= current['step'] - current['activated_steps'][-1]
            )
        elif current['mode']== 'Brace':
            x = wan_attention_cache_forward_brace(
                sa_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['self-attention'],
                ca_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['cross-attention'],
                ffn_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['ffn'],
                e=e,
                x=x,
                distance=current['step'] - current['activated_steps'][-1],
                cache_dic=cache_dic,
                current=current
            )
    
    else:
        raise ValueError(f"Not supported type: {current['type']}")

    return x
