import torch


def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    """
    Update the cache with the fresh tokens.
    """
    step = current["step"]
    layer = current["layer"]
    module = current["module"]
    # Update the cached tokens at the positions

    indices = fresh_indices

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][0].scatter_(
        dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens
    )


def propagation_update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    """
    Update the cache with propagation for ClusCa method.
    
    fresh_indices: torch.Tensor, indices of fresh tokens
    fresh_tokens: torch.Tensor, the fresh tokens to update
    cache_dic: dict, the cache dictionary
    current: dict, current step, layer, and module information
    fresh_attn_map: torch.Tensor, optional attention map (unused)
    """
    step = current["step"]
    layer = current["layer"]
    module = current["module"]

    fresh_tokens = fresh_tokens.to(torch.bfloat16)
    # cluster_info = cache_dic['cluster_info']
    cluster_info = cache_dic["cluster_info"][current["stream"]][current["module"]]
    cluster_indices, cluster_num, k = (
        cluster_info["cluster_indices"],
        cluster_info["cluster_num"],
        cluster_info["k"]
    )
    propagation_ratio = cache_dic["propagation_ratio"]
    dim = fresh_tokens.shape[-1]
    
    propagation_update_cache_compile(
        old_cache_dict=cache_dic["cache"][-1][current["stream"]][layer][module],
        fresh_indices=fresh_indices,
        fresh_tokens=fresh_tokens,
        cluster_indices=cluster_indices,
        cluster_num=cluster_num,
        k=k,
        propagation_ratio=propagation_ratio
    )


@torch.compile
def propagation_update_cache_compile(old_cache_dict, fresh_indices, fresh_tokens, cluster_indices, cluster_num, k, propagation_ratio):
    """
    Compiled version of propagation update for better performance.
    """
    B, N, dim = old_cache_dict[0].shape
    device = old_cache_dict[0].device
    old_cache_dict[0].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)
    
    old_cache = old_cache_dict[0]
    
    # Compute per-cluster means from fresh tokens
    fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)
    sum_per_cluster = torch.zeros((B, cluster_num, dim), device=device, dtype=torch.bfloat16)
    sum_per_cluster.scatter_add_(
        dim=1,
        index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
        src=fresh_tokens
    )
    mean_per_cluster = sum_per_cluster  # only when k == 1
    
    # Propagate cluster means to all tokens in each cluster
    new_cache = mean_per_cluster.gather(1, cluster_indices.unsqueeze(-1).expand(-1, -1, dim))
    old_cache_dict[0] = new_cache * propagation_ratio + old_cache * (1 - propagation_ratio)
