import os
import torch
from pathlib import Path
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass
from transformers import pipeline
from .pipeline_qwenimage import QwenImagePipeline
from .taylor_utils import pipeline_with_taylorseer
from .cache_functions import cache_init

NSFW_THRESHOLD = 0.85
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "qwen-image"


@dataclass
class SamplingOptions:
    prompts: list[str]  # List of prompts
    width: int  # Image width
    height: int  # Image height
    num_steps: int  # Number of sampling steps
    guidance_scale: float  # Guidance scale
    seed: int | None  # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int  # Batch size (batching of prompts)
    model_name: str  # Model name
    output_dir: str  # Output directory
    model_path: str  # Model checkpoint path
    add_sampling_metadata: bool  # Whether to add metadata
    use_nsfw_filter: bool  # Whether to enable NSFW filter
    interval: int  # Cache period length
    max_order: int  # Maximum order of Taylor expansion
    first_enhance: int  # Initial enhancement steps
    negative_prompt: str | None  # Negative prompt for guidance
    taylor_method: str
    hicache_scale: float  # HiCache Hermite scaling factor
    start_index: int  # Global image index offset
    filename: str | None  # Custom output filename


def main(opts: SamplingOptions):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device
        )
    else:
        nsfw_classifier = None

    # Initialize cache
    cache_dic, current = cache_init(method=opts.taylor_method, model_kwargs={
        'num_steps': opts.num_steps,
        'interval': opts.interval,
        'max_order': opts.max_order,
        'first_enhance': opts.first_enhance,
        'hicache_scale': opts.hicache_scale,
    })

    # Load pipeline
    model_path = opts.model_path or os.environ.get("QWEN_IMAGE_MODEL_PATH")
    if not model_path:
        raise ValueError(
            "Qwen-Image model path is not set. Pass --model_path or set QWEN_IMAGE_MODEL_PATH."
        )
    pipe = QwenImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    pipe = pipeline_with_taylorseer(pipe)
    pipe.enable_model_cpu_offload()
    final_output_path = Path(opts.output_dir)
    final_output_path.mkdir(parents=True, exist_ok=True)
    pipe.enable_model_cpu_offload()

    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2 ** 32, (1,)).item()

    total_images = len(opts.prompts) * opts.num_images_per_prompt
    progress_bar = tqdm(
        total=total_images,
        desc="[0/?]",
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    num_prompt_batches = (len(opts.prompts) + opts.batch_size - 1) // opts.batch_size
    idx = opts.start_index
    generated_count = 0

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(opts.prompts))
        batch_prompts = opts.prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            seed = base_seed + idx
            idx += num_prompts_in_batch

            generators = [
                torch.Generator('cpu').manual_seed(int(seed + i))
                for i in range(num_prompts_in_batch)
            ]

            # global index
            global_image_idx = batch_idx * opts.num_images_per_prompt + image_idx

            # Update progress bar desc with current prompt
            prompt_preview = batch_prompts[0][:40] + "..." if len(batch_prompts[0]) > 40 else batch_prompts[0]
            progress_bar.set_description(f"[{generated_count + 1}/{total_images}] {prompt_preview}")

            # Generate images
            result = pipe(
                prompt=batch_prompts,
                negative_prompt=opts.negative_prompt,
                height=opts.height,
                width=opts.width,
                num_inference_steps=opts.num_steps,
                guidance_scale=opts.guidance_scale,
                generator=generators,
                cache_dic=cache_dic,
                current=current,
                image_idx=global_image_idx
            )

            # Handle different return types from pipeline
            images = getattr(result, 'images', None)
            if images is None:
                if isinstance(result, (list, tuple)):
                    images = list(result)
                else:
                    images = [result]

            for i, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    continue

                if opts.use_nsfw_filter and nsfw_classifier is not None:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
                else:
                    nsfw_score = 0.0

                if nsfw_score < NSFW_THRESHOLD:
                    # Add EXIF metadata
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;qwen-image"
                    exif_data[ExifTags.Base.Make] = "Qwen"
                    exif_data[ExifTags.Base.Model] = opts.model_name
                    if opts.add_sampling_metadata and i < len(batch_prompts):
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]

                    # Save image
                    image_idx = idx - num_prompts_in_batch + i
                    if opts.filename:
                        if total_images == 1:
                            filename = final_output_path / opts.filename
                        else:
                            stem, ext = os.path.splitext(opts.filename)
                            filename = final_output_path / f"{stem}_{image_idx}{ext or '.jpg'}"
                    else:
                        filename = final_output_path / f"img_{image_idx}.jpg"
                    img.save(str(filename), exif=exif_data, quality=95, subsampling=0)

                else:
                    print(f"Generated image may contain inappropriate content, skipped.")

                generated_count += 1
                progress_bar.update(1)

    progress_bar.close()
    print(f"Generated {total_images} images in {final_output_path}")


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the Qwen-Image backend.")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="models/qwen_image/prompts/DrawBench200.txt",
        help="Path to the prompt text file.",
    )
    parser.add_argument("--negative_prompt", type=str, default=" ", help="Negative prompt for guidance.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (prompt batching).")
    parser.add_argument("--model_name", type=str, default="qwen-image", choices=["qwen-image"], help="Model name.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save images.")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/brace/premodels/Qwen/Qwen-Image/Qwen/Qwen-Image", help="Path to the Qwen-Image model checkpoint.")
    parser.add_argument(
        "--add_sampling_metadata",
        action="store_true",
        help="Whether to add prompt metadata to images.",
    )
    parser.add_argument("--use_nsfw_filter", action="store_true", help="Enable NSFW filter.")
    parser.add_argument("--interval", type=int, default=7, help="Cache period length.")
    parser.add_argument("--max_order", type=int, default=3, help="Maximum order of Taylor expansion.")
    parser.add_argument("--first_enhance", type=int, default=3, help="Initial enhancement steps.")
    parser.add_argument(
        "--hicache_scale",
        type=float,
        default=0.5,
        help="Scaling factor for HiCache Hermite polynomials.",
    )
    parser.add_argument(
        "--taylor_method",
        type=str,
        default="HiCache",
        choices=["original", "ToCa", "Taylor", "GroupedTaylor", "Delta", "HiCache"],
        help="Choose TaylorSeer/HiCache method.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Global start index for multi-GPU launches.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Custom output filename. If generating a single image, used as-is. For multiple images, index is appended.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N prompts (0 = all).",
    )

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)
    if args.limit > 0:
        prompts = prompts[:args.limit]

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        model_path=args.model_path,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        negative_prompt=args.negative_prompt,
        taylor_method=args.taylor_method,
        hicache_scale=args.hicache_scale,
        start_index=args.start_index,
        filename=args.filename,
    )

    main(opts)