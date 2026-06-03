import os
import re
import time
from dataclasses import dataclass
from glob import iglob
from pathlib import Path

import torch
import torch._dynamo
from fire import Fire
from tqdm import tqdm
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image

NSFW_THRESHOLD = 0.85
PROJECT_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "chipmunk"

torch._dynamo.config.capture_scalar_outputs = True

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


@torch.inference_mode()
def main(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = 50,
    loop: bool = False,
    guidance: float = 3.5,
    cache_mode: str = "chipmunk",
    interval: int = 5,
    max_order: int = 2,
    first_enhance: int = 3,
    hicache_scale: float = 0.6,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    add_sampling_metadata: bool = True,
    trt: bool = False,
    trt_transformer_precision: str = "bf16",
    use_nsfw_filter: bool = False,
    quiet: bool = False,
    **kwargs: dict | None,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        trt: use TensorRT backend for optimized inference
        kwargs: additional arguments for TensorRT support
    """
    import chipmunk.util.config
    chipmunk.util.config.load_from_file(kwargs.get("chipmunk_config", "chipmunk-config.yml"))
    if quiet:
        os.environ["CHIPMUNK_QUIET"] = "1"
    else:
        os.environ.pop("CHIPMUNK_QUIET", None)
    
    if name == 'flux-schnell' or num_steps < 10:
        print("CHIPMUNK: Warning - using Flux-schnell or a low number of steps may result in suboptimal performance. Proceed with caution unless you know what you're doing.")
    prompt = prompt.split("|")
    if len(prompt) == 1:
        prompt = prompt[0]
        additional_prompts: list[str] = []
    else:
        additional_prompts = prompt[1:]
        prompt = prompt[0]

    assert not (
        (additional_prompts is not None) and loop
    ), "Do not provide additional prompts and set loop to True"

    if use_nsfw_filter:
        try:
            nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
        except Exception as e:
            print(f"[WARN] Failed to initialize NSFW classifier, disabling NSFW filter: {e!r}")
            nsfw_classifier = None
    else:
        nsfw_classifier = None

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 128 * (height // 128)
    width = 128 * (width // 128)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    torch._dynamo.config.cache_size_limit = 1 << 32
    torch._dynamo.config.accumulated_cache_size_limit = 1 << 32

    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device=torch_device)
    ae = load_ae(name, device=torch_device)
    from chipmunk.util.config import GLOBAL_CONFIG
    GLOBAL_CONFIG['generation_index'] = 0
    if trt:
        raise ValueError("TensorRT is not supported yet in Chipmunk")

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    total_images = 1 + len(additional_prompts)
    # 始终显示进度条；quiet 仅控制逐张日志与调试输出
    progress_bar = tqdm(total=total_images, desc="Generating images")

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = seed or rng.seed()
        if not quiet:
            print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None

        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        from chipmunk.util.layer_counter import singleton as layer_counter
        # denoise initial noise
        # 统一走 Chipmunk 采样管线：
        # - cache_mode in {"chipmunk","original"} 时，sampling.denoise 内部不会启用 HiCache；
        # - 其他模式（如 HiCache/Taylor）时，sampling.denoise 通过 cache_init/cal_type 驱动 Chipmunk‑FLUX 的缓存执行。
        x = denoise(
            model,
            **inp,
            timesteps=timesteps,
            guidance=opts.guidance,
            cache_mode=cache_mode,
            interval=interval,
            max_order=max_order,
            first_enhance=first_enhance,
            hicache_scale=hicache_scale,
        )



        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        if not quiet:
            print(f"Done in {t1 - t0:.3f}s. Saving {fn}")

        # 使用当前采样使用的 prompt 写入 EXIF，而不是最初的首个 prompt，
        # 以确保多 prompt/multi-GPU 场景下每张图的 ImageDescription 对应正确。
        idx = save_image(nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, opts.prompt)
        progress_bar.update(1)
        
        layer_counter.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        
        GLOBAL_CONFIG['generation_index'] += 1
        if loop:
            if not quiet:
                print("-" * 80)
            opts = parse_prompt(opts)
        elif additional_prompts:
            next_prompt = additional_prompts.pop(0)
            opts.prompt = next_prompt
        else:
            opts = None

    if trt:
        trt_ctx_manager.stop_runtime()
    progress_bar.close()


def app():
    Fire(main)


if __name__ == "__main__":
    app()
