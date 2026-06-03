import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import numpy as np
import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline
from tqdm import tqdm

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, denoise_test_FLOPs
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
    add_sampling_metadata: bool  # Whether to add metadata
    use_nsfw_filter: bool  # Whether to enable NSFW filter
    test_FLOPs: bool  # Whether in FLOPs test mode (no actual image generation)
    cache_mode: str  # Cache mode ('original', 'ToCa', 'Taylor', 'HiCache', 'Delta')
    interval: int  # Cache period length
    max_order: int  # Maximum order of Taylor expansion
    first_enhance: int  # Initial enhancement steps
    collect_features: bool  # Enable feature collection
    feature_layers: list[int]  # Feature collection layer indices (supports multiple layers)
    feature_module: str  # Feature collection module
    feature_stream: str  # Feature collection stream
    skip_decoding: bool  # Skip VAE decoding (feature collection only)
    feature_output_dir: str  # Feature output directory


def main(opts: SamplingOptions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification", model="Falconsai/nsfw_image_detection", device=device
        )
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
    output_name = os.path.join(opts.output_dir, f"img_{{idx}}.jpg")
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    idx = 0  # Image index

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

    # 🔥 修复：添加全局样本计数器用于特征文件命名
    sample_counter = 0

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
                        model, **inp, timesteps=timesteps, guidance=opts.guidance, cache_mode=opts.cache_mode
                    )
                else:
                    # 🔥 配置特征收集（支持多模块一次性收集）
                    feature_config = None
                    if opts.collect_features:
                        feature_config = {
                            "target_layers": opts.feature_layers,
                            "target_modules": ["any"],  # 支持多模块列表
                            "target_streams": ["any"],  # 支持多流列表
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
                        # 🔥 利用现有的特征收集系统
                        enable_feature_collection=opts.collect_features,
                        feature_collection_config=feature_config,
                    )

                    # 🔥 步骤 1: 如果需要，收集特征到内存
                    if opts.collect_features:
                        from flux.taylor_utils import get_collected_features

                        features, metadata = get_collected_features(model._last_cache_dic)
                        # 保存特征数据
                        save_multi_module_features(features, metadata, opts.prompts, opts, sample_counter)

                    # 🔥 步骤 2: 如果需要，进行解码和样本保存
                    if not opts.skip_decoding:
                        # 解码
                        decoded_x = unpack(x.float(), opts.height, opts.width)
                        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                            decoded_x = ae.decode(decoded_x)

                        # 保存样本产物 (图片和 prompt)
                        save_sample_artifacts(decoded_x, batch_prompts, opts, sample_counter, nsfw_classifier)

            # 更新全局样本计数器 (无论哪种模式，一个 prompt 就算一个样本)
            sample_counter += num_prompts_in_batch

            # 更新进度条
            progress_bar.update(num_prompts_in_batch)

    progress_bar.close()


def save_sample_artifacts(decoded_images, prompts, opts: SamplingOptions, start_image_idx, nsfw_classifier):
    """
    保存样本的上下文产物 (图片和 prompt) 到全局的 samples 目录。
    """
    import os
    from PIL import ExifTags, Image

    # 确保全局样本目录存在
    samples_base_dir = os.path.join(opts.feature_output_dir, "samples")
    os.makedirs(samples_base_dir, exist_ok=True)

    # 格式化图像张量
    decoded_images = decoded_images.clamp(-1, 1)
    decoded_images = embed_watermark(decoded_images.float())
    decoded_images = rearrange(decoded_images, "b c h w -> b h w c")

    # 遍历批次中的每个样本
    for i in range(decoded_images.shape[0]):
        current_sample_idx = start_image_idx + i

        # 创建该样本的专属目录
        sample_dir = os.path.join(samples_base_dir, f"sample_{current_sample_idx+1:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # 1. 保存图片
        img_array = decoded_images[i]
        img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().numpy().astype(np.uint8))

        # 可选的 NSFW 过滤
        if opts.use_nsfw_filter and nsfw_classifier:
            nsfw_result = nsfw_classifier(img)
            nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
        else:
            nsfw_score = 0.0

        if nsfw_score < NSFW_THRESHOLD:
            image_path = os.path.join(sample_dir, "image.png")
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            exif_data[ExifTags.Base.Model] = opts.model_name
            exif_data[ExifTags.Base.ImageDescription] = prompts[i]
            img.save(image_path, exif=exif_data, quality=95)
            print(f"   🖼️  样本图片已保存: {image_path}")
        else:
            print(f"   ⚠️  样本 {current_sample_idx+1} 可能包含不当内容，已跳过图片保存。")

        # 2. 保存 Prompt
        prompt_path = os.path.join(sample_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompts[i])
        print(f"   📝 样本 Prompt 已保存: {prompt_path}")


def read_prompts(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_multi_module_features(features, metadata, prompts, opts: SamplingOptions, image_idx):
    """
    保存多模块特征 - 使用参数化的目录结构

    Args:
        features: 嵌套字典 {layer_idx: {module_name: [feature_tensors]}}
        metadata: 嵌套字典 {layer_idx: {module_name: [metadata_dicts]}}
        prompts: 提示列表
        opts: 包含所有采样选项的 SamplingOptions 对象
        image_idx: 图像索引
    """
    import pickle
    import os
    from datetime import datetime

    output_base_dir = opts.feature_output_dir
    if output_base_dir is None:
        output_base_dir = "./features"  # 默认基础目录

    saved_files = []

    print(f"🔄 开始保存参数化特征 (样本 {image_idx})...")

    # 遍历每个层
    for layer_idx, layer_data in features.items():
        # 构建基础路径，不再包含步数信息
        base_path = os.path.join(output_base_dir, opts.model_name, f"l_{layer_idx}")

        # 向后兼容：处理旧格式的单层数据
        if not isinstance(layer_data, dict):
            os.makedirs(base_path, exist_ok=True)
            filename = f"trajectory_sample_{image_idx+1:03d}.pkl"
            filepath = os.path.join(base_path, filename)

            # ... (旧的保存逻辑) ...
            with open(filepath, "wb") as f:
                pickle.dump({}, f)  # 简化
            saved_files.append(filepath)
            continue

        # 处理新格式：遍历每个模块
        for module_name, module_features in layer_data.items():
            # 单流模块（如 layer 28）不创建模块子目录
            if module_name == "total":
                module_output_dir = base_path
            else:
                # 多流模块（如 layer 14）创建模块子目录
                module_output_dir = os.path.join(base_path, f"m_{module_name}")

            os.makedirs(module_output_dir, exist_ok=True)

            # 保存模块特征
            filename = f"trajectory_sample_{image_idx+1:03d}.pkl"
            filepath = os.path.join(module_output_dir, filename)

            module_metadata = metadata.get(layer_idx, {}).get(module_name, [])

            data = {
                # 不再保存 prompt，因为它在全局样本目录中
                "features": module_features,
                "metadata": module_metadata,
                "layer": layer_idx,
                "module": module_name,
                "feature_shape": str(module_features[0].shape) if module_features else "empty",
                "num_timesteps": len(module_features),
                "image_idx": image_idx,
                "timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            saved_files.append(filepath)
            print(f"   ✅ L{layer_idx}-M:{module_name} -> {module_output_dir}")

    print(f"📁 总计保存 {len(saved_files)} 个模块特征文件")
    return saved_files


def save_trajectory_features(features, metadata, prompts, output_dir, image_idx):
    """
    保存轨迹特征到文件 - 兼容旧版本，但推荐使用 save_multi_module_features

    Args:
        features: 特征字典，键为层索引
        metadata: 元数据字典，键为层索引
        prompts: 提示列表
        output_dir: 输出目录
        image_idx: 图像索引
    """
    import pickle
    import os
    from datetime import datetime

    if output_dir is None:
        output_dir = "./golden_trajectories"

    os.makedirs(output_dir, exist_ok=True)

    # 为每个图像批次创建单独的文件
    filename = f"trajectory_batch_{image_idx:03d}.pkl"
    filepath = os.path.join(output_dir, filename)

    data = {
        "features": features,
        "metadata": metadata,
        "prompts": prompts,
        "image_idx": image_idx,
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"特征轨迹已保存到: {filepath}")

    # 统计每层的特征数量
    if isinstance(features, dict):
        for layer_idx, layer_features in features.items():
            if isinstance(layer_features, dict):
                # 新格式：多模块
                for module_name, module_data in layer_features.items():
                    print(f"  Layer {layer_idx} - {module_name}: 收集了 {len(module_data)} 个时间步的特征")
            else:
                # 旧格式：单模块
                print(f"  Layer {layer_idx}: 收集了 {len(layer_features)} 个时间步的特征")
    else:
        # 向后兼容：如果features是列表（旧格式）
        print(f"收集了 {len(features)} 个时间步的特征")

    return filepath


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
            "Delta",
            "collect",
            "ClusCa",
            "Hi-ClusCa",
        ],
        help="Cache mode for denoising.",
    )
    parser.add_argument("--interval", type=int, default=10, help="Cache period length.")
    parser.add_argument("--max_order", type=int, default=1, help="Maximum order of Taylor expansion.")
    parser.add_argument("--first_enhance", type=int, default=3, help="Initial enhancement steps.")
    parser.add_argument("--collect_features", action="store_true", help="Enable feature collection mode.")
    parser.add_argument(
        "--feature_layers",
        type=int,
        nargs="+",
        default=[14],
        help="Feature collection layer indices (supports multiple layers).",
    )
    parser.add_argument(
        "--feature_layer",
        type=int,
        help="Feature collection layer index (legacy, for backward compatibility).",
    )
    parser.add_argument("--feature_module", type=str, default="total", help="Feature collection module.")
    parser.add_argument(
        "--feature_stream", type=str, default="single_stream", help="Feature collection stream."
    )
    parser.add_argument(
        "--skip_decoding", action="store_true", help="Skip VAE decoding (feature collection only)."
    )
    parser.add_argument(
        "--feature_output_dir", type=str, default="./golden_trajectories", help="Feature output directory."
    )

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)

    # Handle legacy feature_layer parameter for backward compatibility
    feature_layers = args.feature_layers
    if args.feature_layer is not None:
        feature_layers = [args.feature_layer]

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
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,
        cache_mode=args.cache_mode,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        collect_features=args.collect_features,
        feature_layers=feature_layers,
        feature_module=args.feature_module,
        feature_stream=args.feature_stream,
        skip_decoding=args.skip_decoding,
        feature_output_dir=args.feature_output_dir,
    )

    main(opts)


if __name__ == "__main__":
    app()
