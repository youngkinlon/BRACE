# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample new images from a pre-trained DiT.
"""
import json
import os
import pickle

import torch

from cache_functions import cache_init

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from brace_utils import *

import time

def load_policy_map(filepath):
    """从磁盘加载策略图文件到内存。"""
    if not os.path.exists(filepath):
        print(f"错误: 策略图文件未找到 -> {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            # static_policy_map 变量现在就是内存中的字典对象
            static_policy_map = pickle.load(f)
            return static_policy_map
    except Exception as e:
        print(f"加载策略图时发生错误: {e}")
        return None

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"/root/autodl-tmp/dit_complete/TaylorSeer-DiT/pretrained_models/DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    #state_dict = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}  # 关键！
    model.load_state_dict(state_dict)
    #model=model.to(dtype=torch.float16)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    class_labels = [289]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    # Classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    model_kwargs['interval'] = args.interval
    model_kwargs['max_order'] = args.max_order
    model_kwargs['test_FLOPs'] = args.test_FLOPs
    # 代表需要 全量计算的步数：
    time_steps=chebyshev_interval_balanced(args.num_sampling_steps)
    model_kwargs['time_steps'] = time_steps

    # 每步计时
    step_times = []
    def timed_forward(*args, **kwargs):
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record()
        out = model.forward_with_cfg(*args, **kwargs)
        step_end.record()
        torch.cuda.synchronize()
        step_times.append(step_start.elapsed_time(step_end))
        return out
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()

    # 采样
    if args.ddim_sample:
        samples = diffusion.ddim_sample_loop(
            timed_forward, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )
    else:
        samples = diffusion.p_sample_loop(
            timed_forward, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )

    total_end.record()
    torch.cuda.synchronize()
    # 总耗时
    total_time = total_start.elapsed_time(total_end) * 0.001
    print(f"\n【总耗时】Total Sampling took {total_time:.3f} seconds")

    # 每步统计
    if step_times:
        step_times_ms = [t for t in step_times]
        print(step_times_ms)
        avg_step = sum(step_times_ms) / len(step_times_ms)
        max_step = max(step_times_ms)
        min_step = min(step_times_ms)
        print(f"【每步耗时】平均: {avg_step:.2f} ms | 最慢: {max_step:.2f} ms | 最快: {min_step:.2f} ms")

        # 跳步率（假设 < 40ms 为 TaylorSeer 跳步）
        skip_rate = sum(1 for t in step_times_ms if t < 40) / len(step_times_ms)
        print(f"【TaylorSeer】跳步率: {skip_rate:.1%} (阈值 <40ms)")
        # 预计 50 步耗时
        print(f"【预测】50 步采样 ≈ {avg_step * 50 / 1000:.2f} 秒")
    # 解码
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    # 保存
    os.makedirs("bary", exist_ok=True)
    log_save_path = "gamma_analysis_log1_n6.json"
    with open(log_save_path, "w", encoding="utf-8") as f:
        json.dump(GLOBAL_ANALYSIS_LOG, f, indent=4)

    save_image(samples, "bary_test/sample404_ruo.png", nrow=4, normalize=True, value_range=(-1, 1))
    print("【完成】图像已保存为 sample.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ddim-sample", action="store_true", default=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--test-FLOPs", action="store_true", default=True)
    args = parser.parse_args()
    import traceback, datetime
    try:
        main(args)
    except Exception as e:
        with open("ddp_error.log", "w", encoding="utf-8") as f:
            f.write(str(datetime.datetime.now()) + "\n")
            traceback.print_exc(file=f)
        raise
    #main(args)

