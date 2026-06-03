import os
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline
from tqdm import tqdm

from flux.sampling import get_noise, get_schedule, prepare, unpack, denoise_test_FLOPs
from flux.ideas import denoise_cache
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85  # NSFW score threshold


@dataclass
class SamplingOptions:
    prompts: list[str]  # List of prompts
    width: int  # Image width
    height: int  # Image height
    num_steps: int  # Number of sampling steps
    guidance: float  # Guidance value
    seed: int | None  # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int  # Batch size (batching of prompts)
    model_name: str  # Model name
    output_dir: str  # Output directory
    start_index: int  # Starting index offset for output numbering
    add_sampling_metadata: bool  # Whether to add metadata
    use_nsfw_filter: bool  # Whether to enable NSFW filter
    test_FLOPs: bool  # Whether in FLOPs test mode (no actual image generation)
    cache_mode: str  # Cache mode ('original', 'ToCa', 'Taylor', 'HiCache', 'Delta', 'collect')
    interval: int  # Cache period length
    max_order: int  # Maximum order of Taylor expansion
    first_enhance: int  # Initial enhancement steps
    hicache_scale: float  # HiCache scaling factor
    # ClusCa parameters
    clusca_fresh_threshold: int  # ClusCa fresh threshold
    clusca_cluster_num: int  # Number of clusters for ClusCa
    clusca_cluster_method: str  # Clustering method (kmeans/kmeans++/random)
    clusca_k: int  # Number of selected fresh tokens per cluster
    clusca_propagation_ratio: float  # Propagation ratio for cluster updates
    # Analytic HiCache (HiCache-Analytic) parameters
    analytic_sigma_alpha: float | None
    analytic_sigma_max: float | None
    analytic_sigma_beta: float | None
    analytic_sigma_eps: float | None
    analytic_sigma_q_quantile: float | None
    analytic_sigma_smooth: float | None
    # Feature collection parameters (enabled when cache_mode='collect')
    feature_layers: list[int]  # Target layers for feature collection
    feature_modules: list[str]  # Target modules for feature collection
    feature_streams: list[str]  # Target streams for feature collection
    skip_decoding: bool  # Skip VAE decoding (feature collection only)
    feature_output_dir: str  # Feature output directory


def main(opts: SamplingOptions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional NSFW classifier（优先使用本地权重，网络/证书异常时自动降级为关闭）
    if opts.use_nsfw_filter:
        try:
            from pathlib import Path

            project_root = Path(__file__).resolve().parents[3]
            nsfw_local_dir = project_root / "weights" / "Falconsai" / "nsfw_image_detection"
            model_id = str(nsfw_local_dir) if nsfw_local_dir.is_dir() else "Falconsai/nsfw_image_detection"
            nsfw_classifier = pipeline(
                "image-classification",
                model=model_id,
                device=device,
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize NSFW classifier, disabling NSFW filter: {e!r}")
            nsfw_classifier = None
    else:
        nsfw_classifier = None

    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if opts.num_steps is None:
        opts.num_steps = 4 if model_name == "flux-schnell" else 50

    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    # Set output directory and index
    # In feature collection mode, save everything to feature_output_dir
    if opts.cache_mode == "collect":
        # Create a timestamp-based subdirectory for this run
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(opts.feature_output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Save images in the same run directory
        output_name = os.path.join(run_dir, "images", "img_{idx}.jpg")
        os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)

        # Save command info and metadata
        save_run_metadata(opts, run_dir)
        save_sampling_config(opts, run_dir)
    else:
        # Normal mode: use specified output_dir
        output_name = os.path.join(opts.output_dir, "img_{idx}.jpg")
        if not os.path.exists(opts.output_dir):
            os.makedirs(opts.output_dir)
        # Save a machine-readable config for reproducibility
        save_sampling_config(opts, opts.output_dir)

    idx = opts.start_index  # Image index offset for numbering

    # Initialize model components
    torch_device = device

    # Load T5 and CLIP models to GPU
    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)

    # Load model to GPU
    model = load_flow_model(model_name, device=torch_device)
    ae = load_ae(model_name, device=torch_device)

    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts

    total_images = len(prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=total_images, desc="Generating images")

    # Compute number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            # Prepare random seed
            seed = base_seed + idx  # Assign a different seed for each image
            idx += num_prompts_in_batch  # Update image index

            # Prepare input
            batch_size = num_prompts_in_batch
            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Prepare prompts
            # batch_prompts is a list containing the prompts in the current batch
            inp = prepare(t5, clip, x, prompt=batch_prompts)
            timesteps = get_schedule(
                opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell")
            )

            # Denoising
            with torch.no_grad():
                if opts.test_FLOPs:
                    x = denoise_test_FLOPs(
                        model,
                        **inp,
                        timesteps=timesteps,
                        guidance=opts.guidance,
                        cache_mode=opts.cache_mode,
                    )
                else:
                    # Configure feature collection (enabled when cache_mode='collect')
                    feature_collection_enabled = opts.cache_mode == "collect"
                    feature_config = None
                    if feature_collection_enabled:
                        if opts.batch_size > 1 or opts.num_images_per_prompt > 1:
                            print(
                                "⚠️ 特征收集当前会在 batch 维度上聚合，"
                                "如需逐图像轨迹分析，建议使用 batch_size=1 且 num_images_per_prompt=1。"
                            )
                        feature_config = {
                            "target_layers": opts.feature_layers,
                            "target_modules": opts.feature_modules,
                            "target_streams": opts.feature_streams,
                        }

                    x = denoise_cache(
                        model,
                        **inp,
                        timesteps=timesteps,
                        guidance=opts.guidance,
                        cache_mode=opts.cache_mode,
                        interval=opts.interval,
                        max_order=opts.max_order,
                        first_enhance=opts.first_enhance,
                        hicache_scale=opts.hicache_scale,
                        # ClusCa parameters
                        clusca_fresh_threshold=opts.clusca_fresh_threshold,
                        clusca_cluster_num=opts.clusca_cluster_num,
                        clusca_cluster_method=opts.clusca_cluster_method,
                        clusca_k=opts.clusca_k,
                        clusca_propagation_ratio=opts.clusca_propagation_ratio,
                         analytic_sigma_alpha=opts.analytic_sigma_alpha,
                         analytic_sigma_max=opts.analytic_sigma_max,
                         analytic_sigma_beta=opts.analytic_sigma_beta,
                         analytic_sigma_eps=opts.analytic_sigma_eps,
                         analytic_sigma_q_quantile=opts.analytic_sigma_q_quantile,
                         analytic_sigma_smooth=opts.analytic_sigma_smooth,
                        # Feature collection parameters
                        enable_feature_collection=feature_collection_enabled,
                        feature_collection_config=feature_config,
                    )
                    # x = search_denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, interval=opts.interval, max_order=opts.max_order, first_enhance=opts.first_enhance)

                # Handle feature collection
                if feature_collection_enabled:
                    from flux.taylor_utils import get_collected_features

                    features, metadata = get_collected_features(model._last_cache_dic)
                    # Save feature data to the same run directory
                    current_run_dir = None
                    if opts.cache_mode == "collect":
                        # Extract run_dir from output_name path
                        current_run_dir = os.path.dirname(
                            os.path.dirname(output_name)
                        )  # Go up from /images/img_x.jpg
                    save_collected_features(
                        features, metadata, batch_prompts, opts, idx - num_prompts_in_batch, current_run_dir
                    )

                # Decode latent variables (skip if only collecting features)
                if not opts.skip_decoding:
                    x = unpack(x.float(), opts.height, opts.width)
                    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                        x = ae.decode(x)

            # Convert to PIL format and save (skip if only collecting features)
            if not opts.skip_decoding:
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x, "b c h w -> b h w c")

                for i in range(batch_size):
                    img_array = x[i]
                    img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().numpy().astype(np.uint8))

                    # Optional NSFW filtering
                    if opts.use_nsfw_filter:
                        nsfw_result = nsfw_classifier(img)
                        nsfw_score = next(
                            (res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0
                        )
                    else:
                        nsfw_score = 0.0  # If the filter is not enabled, assume safe

                    if nsfw_score < NSFW_THRESHOLD:
                        exif_data = Image.Exif()
                        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                        exif_data[ExifTags.Base.Model] = model_name
                        if opts.add_sampling_metadata:
                            exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                        # Save image
                        fn = output_name.format(idx=idx - num_prompts_in_batch + i)
                        img.save(fn, exif=exif_data, quality=95, subsampling=0)
                    else:
                        print("Generated image may contain inappropriate content, skipped.")

                    progress_bar.update(1)
            else:
                # If skipping decoding, still update progress bar
                for i in range(batch_size):
                    progress_bar.update(1)

    progress_bar.close()


def read_prompts(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_run_metadata(opts: SamplingOptions, run_dir: str):
    """
    保存运行的元数据和命令信息

    Args:
        opts: 采样选项
        run_dir: 运行目录
    """
    import json
    import sys
    from datetime import datetime

    # Save command information
    command_info = {
        "timestamp": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "working_directory": os.getcwd(),
        "config": {
            "cache_mode": opts.cache_mode,
            "feature_layers": opts.feature_layers,
            "feature_modules": opts.feature_modules,
            "feature_streams": opts.feature_streams,
            "model_name": opts.model_name,
            "width": opts.width,
            "height": opts.height,
            "num_steps": opts.num_steps,
            "guidance": opts.guidance,
            "seed": opts.seed,
            "num_images_per_prompt": opts.num_images_per_prompt,
            "batch_size": opts.batch_size,
            "interval": opts.interval if opts.cache_mode != "collect" else "auto (每步刷新)",
            "max_order": opts.max_order if opts.cache_mode != "collect" else 0,
            "first_enhance": opts.first_enhance if opts.cache_mode != "collect" else opts.num_steps,
            "hicache_scale": opts.hicache_scale if opts.cache_mode != "collect" else "unused",
            "skip_decoding": opts.skip_decoding,
            # 额外信息用于collect模式
            "actual_cache_behavior": "collect模式: 基于original实现，每步完整计算，无缓存优化"
            if opts.cache_mode == "collect"
            else f"{opts.cache_mode}模式缓存",
            "command_line_note": "注意: 命令行的interval/max_order等参数在collect模式下被自动覆盖"
            if opts.cache_mode == "collect"
            else None,
        },
    }

    # Save command info as JSON
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(command_info, f, indent=2, ensure_ascii=False)

    # Save command as shell script for reproduction
    with open(os.path.join(run_dir, "reproduce_command.sh"), "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# 重现此次运行的命令\n")
        f.write("# 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        # Add environment variables
        f.write("# 设置环境变量\n")
        f.write('export FLUX_DEV="/path/to/FLUX.1-dev/flux1-dev.safetensors"\n')
        f.write('export AE="/path/to/FLUX.1-dev/ae.safetensors"\n\n')

        # Add the original command
        f.write("# 运行命令\n")
        f.write(" ".join(sys.argv) + "\n")

    # Copy prompt file to run directory
    try:
        prompt_file_path = sys.argv[sys.argv.index("--prompt_file") + 1]
        if os.path.exists(prompt_file_path):
            import shutil

            prompt_file_name = os.path.basename(prompt_file_path)
            shutil.copy2(prompt_file_path, os.path.join(run_dir, f"prompts_{prompt_file_name}"))
    except (ValueError, IndexError):
        # If --prompt_file not found in sys.argv, skip copying
        pass

    print(f"📋 运行信息已保存到: {run_dir}")


def save_sampling_config(opts: SamplingOptions, out_dir: str):
    """Dump all key sampling parameters to a config.json in the output directory.

    This complements metadata and enables reproducible evaluation by capturing
    cache settings (interval, order, scales), sampler settings, and model info.
    """
    import json
    import sys
    from datetime import datetime

    os.makedirs(out_dir, exist_ok=True)

    # Build a structured config
    cfg = {
        "timestamp": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "working_directory": os.getcwd(),
        "mode": opts.cache_mode,
        "sampler": {
            "width": int(opts.width),
            "height": int(opts.height),
            "num_steps": int(opts.num_steps),
            "guidance": float(opts.guidance),
            "seed": int(opts.seed) if isinstance(opts.seed, int) else opts.seed,
            "num_images_per_prompt": int(opts.num_images_per_prompt),
            "batch_size": int(opts.batch_size),
            "start_index": int(opts.start_index),
        },
        "model": {
            "name": opts.model_name,
        },
        "cache": {
            "interval": int(opts.interval),
            "max_order": int(opts.max_order),
            "first_enhance": int(opts.first_enhance),
            "hicache_scale": float(opts.hicache_scale),
            "analytic_sigma_alpha": float(opts.analytic_sigma_alpha)
            if opts.analytic_sigma_alpha is not None
            else None,
            "analytic_sigma_max": float(opts.analytic_sigma_max)
            if opts.analytic_sigma_max is not None
            else None,
            "analytic_sigma_beta": float(opts.analytic_sigma_beta)
            if opts.analytic_sigma_beta is not None
            else None,
            "analytic_sigma_eps": float(opts.analytic_sigma_eps)
            if opts.analytic_sigma_eps is not None
            else None,
            "analytic_sigma_q_quantile": float(opts.analytic_sigma_q_quantile)
            if opts.analytic_sigma_q_quantile is not None
            else None,
            "analytic_sigma_smooth": float(opts.analytic_sigma_smooth)
            if opts.analytic_sigma_smooth is not None
            else None,
        },
        "clusca": {
            "fresh_threshold": int(opts.clusca_fresh_threshold),
            "cluster_num": int(opts.clusca_cluster_num),
            "cluster_method": opts.clusca_cluster_method,
            "k": int(opts.clusca_k),
            "propagation_ratio": float(opts.clusca_propagation_ratio),
        },
        "prompts": {
            "count": len(opts.prompts),
        },
    }

    # Persist to JSON
    dst = os.path.join(out_dir, "config.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # Silent by default (avoid noisy logs under multi-gpu), caller can inspect file


def save_collected_features(features, metadata, prompts, opts: SamplingOptions, image_idx, run_dir=None):
    """
    保存收集的特征数据

    Args:
        features: 嵌套字典 {layer_idx: {module_name: [feature_tensors]}}
        metadata: 嵌套字典 {layer_idx: {module_name: [metadata_dicts]}}
        prompts: 提示列表
        opts: 包含所有采样选项的 SamplingOptions 对象
        image_idx: 图像索引
        run_dir: 运行目录（如果在collect模式下）
    """
    import pickle
    import os
    from datetime import datetime

    if run_dir:
        # In collect mode: save to run directory
        output_base_dir = os.path.join(run_dir, "features")
    else:
        # Normal mode: use feature_output_dir
        output_base_dir = opts.feature_output_dir

    os.makedirs(output_base_dir, exist_ok=True)

    saved_files = []

    print(f"🔄 开始保存特征 (样本 {image_idx})...")

    # 遍历每个层
    for layer_idx, layer_data in features.items():
        # 构建基础路径
        base_path = os.path.join(output_base_dir, opts.model_name, f"layer_{layer_idx}")

        # 处理每个模块
        for module_name, module_features in layer_data.items():
            # 创建模块目录
            if module_name == "total":
                module_output_dir = base_path
            else:
                module_output_dir = os.path.join(base_path, f"module_{module_name}")

            os.makedirs(module_output_dir, exist_ok=True)

            # 保存模块特征
            filename = f"features_sample_{image_idx+1:03d}.pkl"
            filepath = os.path.join(module_output_dir, filename)

            module_metadata = metadata.get(layer_idx, {}).get(module_name, [])

            data = {
                "features": module_features,
                "metadata": module_metadata,
                "prompts": prompts,
                "layer": layer_idx,
                "module": module_name,
                "feature_shape": str(module_features[0].shape) if module_features else "empty",
                "num_timesteps": len(module_features),
                "image_idx": image_idx,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "cache_mode": opts.cache_mode,
                    "interval": opts.interval if opts.cache_mode != "collect" else "auto (每步刷新)",
                    "max_order": opts.max_order if opts.cache_mode != "collect" else 0,
                    "first_enhance": opts.first_enhance
                    if opts.cache_mode != "collect"
                    else len(metadata.get(layer_idx, {}).get(module_name, [])),
                    "hicache_scale": opts.hicache_scale if opts.cache_mode != "collect" else "unused",
                    "actual_mode": "collect (基于original模式 + 特征收集)"
                    if opts.cache_mode == "collect"
                    else opts.cache_mode,
                    "feature_collection_note": "收集模式：每步都进行完整计算，无缓存加速，专注特征提取"
                    if opts.cache_mode == "collect"
                    else None,
                },
            }

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            saved_files.append(filepath)
            print(f"   ✅ Layer {layer_idx}, Module {module_name} -> {filepath}")

    print(f"📁 总计保存 {len(saved_files)} 个特征文件")
    return saved_files


def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt text file.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of sampling steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (prompt batching).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell"],
        help="Model name.",
    )
    parser.add_argument("--output_dir", type=str, default="./samples", help="Directory to save images.")
    parser.add_argument(
        "--add_sampling_metadata", action="store_true", help="Whether to add prompt metadata to images."
    )
    parser.add_argument("--use_nsfw_filter", action="store_true", help="Enable NSFW filter.")
    parser.add_argument("--test_FLOPs", action="store_true", help="Test inference computation cost.")
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="original",
        choices=[
            "original",
            "ToCa",
            "Taylor",
            "Taylor-Scaled",
            "HiCache",
            "HiCache-Analytic",
            "Delta",
            "collect",
            "ClusCa",
            "Hi-ClusCa",
        ],
        help="Cache mode for denoising.",
    )
    parser.add_argument("--interval", type=int, default=10, help="Cache period length.")
    parser.add_argument("--max_order", type=int, default=5, help="Maximum order of Taylor expansion.")
    parser.add_argument("--first_enhance", type=int, default=5, help="Initial enhancement steps.")
    parser.add_argument("--hicache_scale", type=float, default=1.0, help="HiCache scaling factor.")
    # ClusCa arguments
    parser.add_argument(
        "--clusca_fresh_threshold",
        type=int,
        default=5,
        help="ClusCa fresh threshold.",
    )
    parser.add_argument(
        "--clusca_cluster_num",
        type=int,
        default=16,
        help="Number of clusters for ClusCa.",
    )
    parser.add_argument(
        "--clusca_cluster_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "kmeans++", "random"],
        help="Clustering method for ClusCa.",
    )
    parser.add_argument(
        "--clusca_k",
        type=int,
        default=1,
        help="Number of selected fresh tokens per cluster.",
    )
    parser.add_argument(
        "--clusca_propagation_ratio",
        type=float,
        default=0.005,
        help="Propagation ratio for cluster updates.",
    )
    # Analytic HiCache (HiCache-Analytic) arguments
    parser.add_argument(
        "--analytic_sigma_alpha",
        type=float,
        default=None,
        help="Alpha factor for analytic sigma in HiCache-Analytic (default 1.28 → sigma≈0.9 when q≈1).",
    )
    parser.add_argument(
        "--analytic_sigma_max",
        type=float,
        default=None,
        help="Upper bound for analytic sigma in HiCache-Analytic (default 1.0).",
    )
    parser.add_argument(
        "--analytic_sigma_beta",
        type=float,
        default=None,
        help="EMA smoothing factor beta for analytic sigma statistics (default 0.01). "
        "Set to 0 to disable online updates and use the closed-form with q=1.",
    )
    parser.add_argument(
        "--analytic_sigma_eps",
        type=float,
        default=None,
        help="Epsilon added to the denominator in analytic sigma formula (default 1e-6).",
    )
    parser.add_argument(
        "--analytic_sigma_q_quantile",
        type=float,
        default=None,
        help="Optional quantile (e.g., 0.95) for robust q estimation; if unset, use mean.",
    )
    parser.add_argument(
        "--analytic_sigma_smooth",
        type=float,
        default=None,
        help="Gamma for log-domain EMA smoothing of sigma (0 disables smoothing).",
    )
    # Feature collection arguments (enabled when cache_mode='collect')
    parser.add_argument(
        "--feature_layers",
        type=int,
        nargs="+",
        default=[14],
        help="Target layers for feature collection (supports multiple layers).",
    )
    parser.add_argument(
        "--feature_modules",
        type=str,
        nargs="+",
        default=["any"],
        help="Target modules for feature collection.",
    )
    parser.add_argument(
        "--feature_streams",
        type=str,
        nargs="+",
        default=["any"],
        help="Target streams for feature collection.",
    )
    parser.add_argument(
        "--skip_decoding", action="store_true", help="Skip VAE decoding (feature collection only)."
    )
    parser.add_argument(
        "--feature_output_dir", type=str, default="./features", help="Feature output directory."
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting index offset for img_*.jpg numbering."
    )

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        start_index=args.start_index,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,
        cache_mode=args.cache_mode,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        hicache_scale=args.hicache_scale,
        # ClusCa parameters
        clusca_fresh_threshold=args.clusca_fresh_threshold,
        clusca_cluster_num=args.clusca_cluster_num,
        clusca_cluster_method=args.clusca_cluster_method,
        clusca_k=args.clusca_k,
        clusca_propagation_ratio=args.clusca_propagation_ratio,
        analytic_sigma_alpha=args.analytic_sigma_alpha,
        analytic_sigma_max=args.analytic_sigma_max,
        analytic_sigma_beta=args.analytic_sigma_beta,
        analytic_sigma_eps=args.analytic_sigma_eps,
        analytic_sigma_q_quantile=args.analytic_sigma_q_quantile,
        analytic_sigma_smooth=args.analytic_sigma_smooth,
        # Feature collection parameters (enabled when cache_mode='collect')
        feature_layers=args.feature_layers,
        feature_modules=args.feature_modules,
        feature_streams=args.feature_streams,
        skip_decoding=args.skip_decoding,
        feature_output_dir=args.feature_output_dir,
    )

    main(opts)


if __name__ == "__main__":
    app()
