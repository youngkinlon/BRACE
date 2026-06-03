import torch
from ..model import Flux
from torch import Tensor
from ..modules.cache_functions import cache_init


def denoise_cache(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # cache mode
    cache_mode: str = "Taylor",
    # cache parameters
    interval: int = 6,
    max_order: int = 1,
    first_enhance: int = 3,
    hicache_scale: float = 0.5,
    # ClusCa parameters
    clusca_fresh_threshold: int | None = None,
    clusca_cluster_num: int | None = None,
    clusca_cluster_method: str | None = None,
    clusca_k: int | None = None,
    clusca_propagation_ratio: float | None = None,
    # 🔥 新增：特征收集参数
    enable_feature_collection: bool = False,
    feature_collection_config: dict = None,
):
    clusca_kwargs = None
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        clusca_kwargs = {
            "clusca_fresh_threshold": clusca_fresh_threshold,
            "clusca_cluster_num": clusca_cluster_num,
            "clusca_cluster_method": clusca_cluster_method,
            "clusca_k": clusca_k,
            "clusca_propagation_ratio": clusca_propagation_ratio,
        }
        clusca_kwargs = {k: v for k, v in clusca_kwargs.items() if v is not None}

    # init cache with specified mode and parameters
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        model_kwargs = clusca_kwargs
    else:
        model_kwargs = None
    cache_dic, current = cache_init(
        timesteps,
        model_kwargs=model_kwargs,
        mode=cache_mode,
        interval=interval,
        max_order=max_order,
        first_enhance=first_enhance,
        hicache_scale=hicache_scale,
    )

    # 🔥 新增：配置特征收集
    if enable_feature_collection:
        cache_dic["enable_feature_collection"] = True
        cache_dic["feature_collection_config"] = feature_collection_config or {
            "target_layer": 14,
            "target_module": "total",
            "target_stream": "single_stream",
        }

    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    current["step"] = 0
    current["num_steps"] = len(timesteps) - 1
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        current["t"] = t_curr
        # print(t_curr)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            cache_dic=cache_dic,
            current=current,
            guidance=guidance_vec,
        )
        # print(img.shape)
        img = img + (t_prev - t_curr) * pred
        current["step"] += 1

    # 🔥 新增：将cache_dic存储到模型中以便后续访问
    model._last_cache_dic = cache_dic

    return img
