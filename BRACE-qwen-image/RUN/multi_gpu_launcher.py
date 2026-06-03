#!/usr/bin/env python3

"""Launch multiple `sample.sh` runs across available GPUs.

The script splits the prompt list into shards (one per GPU) and spawns a
separate `sample.sh` process for each shard with an isolated prompt file and
output directory.  Additional arguments after `--` are forwarded verbatim to
`sample.sh`, except for options the launcher manages itself (`--mode`,
`--prompt_file`, `--output_dir`).

Example:

    python RUN/multi_gpu_launcher.py \
        --mode HiCache \
        --gpus 0,1 \
        --prompt-file resources/prompts/prompt.txt \
        --base-output-dir results/hicache_multi \
        -- --limit 8 --interval 6 --max_order 2 --hicache_scale 0.6

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SH = PROJECT_ROOT / "scripts" / "sample.sh"
RUN_BACKEND_SH = PROJECT_ROOT / "scripts" / "run_backend.sh"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "resources" / "prompts" / "prompt.txt"
CHIPMUNK_EXAMPLE_DIR = PROJECT_ROOT / "models" / "chipmunk" / "examples" / "flux"


@dataclass
class ShardJob:
    gpu: str
    prompts: List[str]
    start_index: int
    prompt_file: Path
    output_dir: Path
    process: subprocess.Popen | None = None
    full_output_dir: Path | None = None
    placeholder_image: Path | None = None


IMG_PATTERN = re.compile(r"img_(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch sample.sh or backend runners on multiple GPUs in parallel")
    parser.add_argument(
        "--backend",
        default="flux",
        choices=["flux", "qwen-image", "chipmunk"],
        help="Backend to run for each shard (flux uses scripts/sample.sh)",
    )
    parser.add_argument("--mode", required=True, help="Cache mode to pass to sample.sh")
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Path to the full prompt list (default: %(default)s)",
    )
    parser.add_argument(
        "--full-prompt-file",
        help="Path to the master prompt list for offset/manifest (defaults to --prompt-file)",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU ids to use (e.g. '0,1,3'). If omitted, tries CUDA_VISIBLE_DEVICES or torch.cuda.device_count().",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs to use starting from 0 when --gpus is not provided",
    )
    parser.add_argument(
        "--base-output-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Base directory for outputs. Each GPU shard gets its own subfolder.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional name appended to output directories. Defaults to timestamp.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated prompt shards under RUN/tmp_multi_gpu_*",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional JSON file path to record the final merged output directory.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Global start index offset for numbering/resume.",
    )
    parser.add_argument(
        "--chipmunk-param-tag",
        help="Relative output subdirectory for chipmunk backend runs.",
    )
    parser.add_argument(
        "sample_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to sample.sh (prefix with --).",
    )
    return parser.parse_args()


def detect_gpus(gpu_arg: str | None, num_gpus: int | None) -> List[str]:
    if gpu_arg:
        return [gpu.strip() for gpu in gpu_arg.split(",") if gpu.strip()]

    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        tokens = [token.strip() for token in env.split(",") if token.strip()]
        if tokens:
            return tokens

    if num_gpus is not None:
        return [str(i) for i in range(num_gpus)]

    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:  # pragma: no cover - torch optional
        count = 0

    if count > 0:
        return [str(i) for i in range(count)]

    # fallback to single GPU 0
    return ["0"]


def read_prompts(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"Prompt file '{path}' is empty")
    return prompts


def split_prompts(prompts: List[str], max_shards: int) -> List[tuple[int, List[str]]]:
    if max_shards <= 0:
        raise ValueError("Number of shards must be positive")

    total = len(prompts)
    if total == 0:
        raise ValueError("Prompt list is empty")

    shards = min(max_shards, total)
    base = total // shards
    remainder = total % shards

    result: List[tuple[int, List[str]]] = []
    start = 0
    for shard_idx in range(shards):
        size = base + (1 if shard_idx < remainder else 0)
        end = start + size
        chunk = prompts[start:end]
        if chunk:
            result.append((start, chunk))
        start = end

    return result


def sanitize_sample_args(sample_args: Iterable[str]) -> List[str]:
    args = list(sample_args)
    if args and args[0] == "--":
        args = args[1:]

    managed_flags = {"--mode", "--prompt_file", "--prompt", "--output_dir", "--start_index"}
    for token in args:
        for flag in managed_flags:
            if token == flag or token.startswith(flag + "="):
                raise ValueError(f"Launcher manages '{flag}'; remove it from forwarded arguments: {args}")
    return args


def build_output_dir(stage_root: Path, gpu: str) -> Path:
    return stage_root / f"gpu{gpu}"


def write_prompt_shard(prompts: List[str], directory: Path, gpu: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shard_path = directory / f"prompts_gpu{gpu}.txt"
    with shard_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(prompts))
        f.write("\n")
    return shard_path


def normalize_chipmunk_args(extra_args: List[str]) -> List[str]:
    """Ensure chipmunk-specific arguments (like config paths) are absolute."""
    normalized: List[str] = []
    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if token in {"--chipmunk_config", "--chipmunk-config"}:
            if i + 1 >= len(extra_args):
                raise ValueError(f"{token} requires a path argument")
            path_token = extra_args[i + 1]
            resolved = Path(path_token)
            if not resolved.is_absolute():
                resolved = (PROJECT_ROOT / resolved).resolve()
            normalized.append("--chipmunk-config")
            normalized.append(str(resolved))
            i += 2
            continue
        if token.startswith("--chipmunk_config=") or token.startswith("--chipmunk-config="):
            option, path_token = token.split("=", 1)
            resolved = Path(path_token)
            if not resolved.is_absolute():
                resolved = (PROJECT_ROOT / resolved).resolve()
            normalized.append("--chipmunk-config=" + str(resolved))
            i += 1
            continue
        normalized.append(token)
        i += 1
    return normalized


def setup_chipmunk_output(job: ShardJob, mode_label: str, param_tag: str, dry_run: bool) -> tuple[Path, Path | None]:
    if not param_tag:
        raise ValueError("chipmunk backend requires --chipmunk-param-tag")

    final_dir = job.output_dir / mode_label / param_tag
    placeholder: Path | None = None
    if not dry_run:
        final_dir.mkdir(parents=True, exist_ok=True)
        marker_path = job.output_dir / ".full_output_dir"
        marker_path.write_text(str(final_dir.resolve()), encoding="utf-8")
        if job.start_index > 0:
            placeholder = final_dir / f"img_{job.start_index - 1}.jpg"
            placeholder.touch(exist_ok=True)
    return final_dir, placeholder


def launch_process(
    gpu: str,
    prompt_file: Path,
    mode: str,
    output_dir: Path,
    start_index: int,
    extra_args: List[str],
    dry_run: bool,
    backend: str,
    prompts: List[str] | None = None,
) -> subprocess.Popen | None:
    run_cwd = PROJECT_ROOT
    if backend == "flux":
        cmd = [
            "bash",
            str(SAMPLE_SH),
            "--mode",
            mode,
            "--prompt_file",
            str(prompt_file),
            "--output_dir",
            str(output_dir),
            "--start_index",
            str(start_index),
        ] + extra_args
    elif backend == "qwen-image":
        cmd = [
            "bash",
            str(RUN_BACKEND_SH),
            "--backend",
            "qwen-image",
            "--",
            "--prompt_file",
            str(prompt_file),
            "--output_dir",
            str(output_dir),
            "--start_index",
            str(start_index),
        ]
        has_taylor_flag = any(
            arg == "--taylor_method" or arg.startswith("--taylor_method=") for arg in extra_args
        )
        if not has_taylor_flag:
            cmd += ["--taylor_method", mode]
        cmd += extra_args
    elif backend == "chipmunk":
        if not prompts:
            raise ValueError("Chipmunk backend requires prompt shards.")
        extra_args = normalize_chipmunk_args(extra_args)
        joined_prompts = "|".join(prompts)
        python_bin = sys.executable
        cmd = [
            python_bin,
            "-m",
            "flux.cli",
            "--prompt",
            joined_prompts,
            "--output_dir",
            str(output_dir),
        ] + extra_args
        run_cwd = CHIPMUNK_EXAMPLE_DIR
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTHON_BIN", sys.executable)
    if backend == "chipmunk":
        # 确保优先使用 Chipmunk 示例里的 flux 包，而不是仓库根目录安装的 HiCache-Flux 版本
        chipmunk_src = str(CHIPMUNK_EXAMPLE_DIR / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = chipmunk_src if not existing else chipmunk_src + os.pathsep + existing

    print("[LAUNCH] GPU", gpu, "->", " ".join(cmd))
    if dry_run:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, cwd=run_cwd, env=env)


def collect_numbered_files(directory: Path, pattern: re.Pattern) -> List[tuple[int, Path]]:
    items: List[tuple[int, Path]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            items.append((int(match.group(1)), path))
    items.sort(key=lambda item: item[0])
    return items


def aggregate_outputs(jobs: List[ShardJob], aggregated_root: Path, prompts: List[str]) -> Path | None:
    if not jobs:
        return None

    if aggregated_root.exists():
        shutil.rmtree(aggregated_root)
    aggregated_root.mkdir(parents=True, exist_ok=True)

    prompt_path = aggregated_root / "prompts.txt"
    prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    manifest: List[dict] = []
    rel_subpaths: set[Path] = set()

    for job in sorted(jobs, key=lambda j: j.start_index):
        marker = job.output_dir / ".full_output_dir"
        if not marker.exists():
            raise FileNotFoundError(f"Missing .full_output_dir marker in {job.output_dir}")

        full_output_dir = Path(marker.read_text(encoding="utf-8").strip()).resolve()
        if not full_output_dir.exists():
            raise FileNotFoundError(f"Expected output directory '{full_output_dir}' not found")

        job.full_output_dir = full_output_dir

        try:
            rel_subpath = full_output_dir.relative_to(job.output_dir)
        except ValueError:
            rel_subpath = Path(full_output_dir.name)

        rel_subpaths.add(rel_subpath)
        dest_dir = aggregated_root / rel_subpath
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy numbered images
        for index, src_path in collect_numbered_files(full_output_dir, IMG_PATTERN):
            dest_path = dest_dir / src_path.name
            if dest_path.exists():
                raise FileExistsError(f"Duplicate output file detected: {dest_path}")
            shutil.copy2(src_path, dest_path)
            prompt_text = prompts[index] if index < len(prompts) else None
            manifest.append(
                {
                    "index": index,
                    "prompt": prompt_text,
                    "source_gpu": job.gpu,
                    "source_dir": str(full_output_dir),
                    "source_image": str(src_path),
                    "dest_image": str(dest_path),
                }
            )

        # Copy logs if present
        logs_dir = full_output_dir / "logs"
        if logs_dir.is_dir():
            dest_logs_dir = dest_dir / "logs"
            shutil.copytree(logs_dir, dest_logs_dir, dirs_exist_ok=True)

        # Copy any other auxiliary files/directories if they haven't been copied yet
        for item in full_output_dir.iterdir():
            if item.name in {"logs"}:
                continue
            if item.is_file() and IMG_PATTERN.fullmatch(item.name):
                continue
            dest_item = dest_dir / item.name
            if dest_item.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest_item)
            else:
                shutil.copy2(item, dest_item)

    manifest.sort(key=lambda entry: entry["index"])
    manifest_path = aggregated_root / "merged_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(rel_subpaths) == 1:
        return aggregated_root / next(iter(rel_subpaths))
    return aggregated_root


def main() -> int:
    args = parse_args()

    if args.backend == "chipmunk" and not args.chipmunk_param_tag:
        raise ValueError("Chipmunk backend requires --chipmunk-param-tag to organize outputs.")

    gpu_list = detect_gpus(args.gpus, args.num_gpus)
    if not gpu_list:
        raise RuntimeError("No GPU devices available")

    prompt_path = Path(args.prompt_file)
    full_prompt_path = Path(args.full_prompt_file) if args.full_prompt_file else prompt_path
    master_prompts = read_prompts(full_prompt_path)
    prompt_list = read_prompts(prompt_path)
    start_offset = max(int(args.start_offset or 0), 0)
    if start_offset >= len(master_prompts):
        raise ValueError(
            f"Start offset {start_offset} exceeds available prompts ({len(master_prompts)})."
        )
    if args.full_prompt_file:
        if not prompt_list:
            raise ValueError("Prompt file is empty after preprocessing.")
        active_prompts = prompt_list
    else:
        if start_offset >= len(prompt_list):
            raise ValueError("Start offset exceeds prompt file length.")
        active_prompts = prompt_list[start_offset:]
    if not active_prompts:
        raise ValueError("No prompts remain after applying prompt filtering.")
    shard_specs = split_prompts(active_prompts, len(gpu_list))

    if len(shard_specs) < len(gpu_list):
        unused_gpus = gpu_list[len(shard_specs) :]
        if unused_gpus:
            print(f"[INFO] GPUs {', '.join(unused_gpus)} idle (no prompts assigned)")

    extra_args = sanitize_sample_args(args.sample_args)

    base_output_dir = Path(args.base_output_dir).resolve()
    default_root = (PROJECT_ROOT / "results").resolve()
    if base_output_dir == default_root:
        backend_root = args.backend.replace("-", "_")
        base_output_dir = default_root / backend_root
    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = args.backend.replace("-", "_")
    mode_label = args.mode.lower()
    stage_label = mode_label if args.backend in {"flux", "chipmunk"} else f"{backend_tag}_{mode_label}"
    stage_root = base_output_dir / ".multi_gpu_tmp" / f"{stage_label}_{timestamp_label}"
    stage_root.mkdir(parents=True, exist_ok=True)
    final_mode_dir_name = stage_label if not args.run_name else f"{stage_label}_{args.run_name}"
    aggregated_root = stage_root / "merged"
    target_mode_dir = base_output_dir / final_mode_dir_name
    tmp_root = Path(tempfile.mkdtemp(prefix="tmp_multi_gpu_", dir=PROJECT_ROOT / "RUN"))

    exit_code = 0
    jobs: List[ShardJob] = []
    final_output_path: Path | None = None
    success = False
    try:
        for gpu, (start_index, shard_prompts) in zip(gpu_list, shard_specs):
            shard_file = write_prompt_shard(shard_prompts, tmp_root, gpu)
            shard_output = build_output_dir(stage_root, gpu)
            global_start = start_offset + start_index

            job = ShardJob(
                gpu=gpu,
                prompts=shard_prompts,
                start_index=global_start,
                prompt_file=shard_file,
                output_dir=shard_output,
            )

            backend_output_dir = shard_output
            if args.backend == "chipmunk":
                full_output_dir, placeholder = setup_chipmunk_output(
                    job,
                    mode_label=mode_label,
                    param_tag=args.chipmunk_param_tag,
                    dry_run=args.dry_run,
                )
                job.full_output_dir = full_output_dir
                job.placeholder_image = placeholder
                backend_output_dir = full_output_dir

            job.process = launch_process(
                gpu=gpu,
                prompt_file=shard_file,
                mode=args.mode,
                output_dir=backend_output_dir,
                start_index=global_start,
                extra_args=extra_args,
                dry_run=args.dry_run,
                backend=args.backend,
                prompts=shard_prompts,
            )
            jobs.append(job)

        if args.dry_run:
            print("[DRY-RUN] No commands executed.")
            success = True
            return 0

        for job in jobs:
            if job.process is None:
                continue
            ret = job.process.wait()
            if ret != 0:
                print(f"[ERROR] Process on GPU {job.gpu} exited with code {ret}", file=sys.stderr)
                exit_code = ret if exit_code == 0 else exit_code

        if args.backend == "chipmunk":
            for job in jobs:
                placeholder = job.placeholder_image
                if placeholder and placeholder.exists():
                    try:
                        placeholder.unlink(missing_ok=True)
                    except Exception as cleanup_err:
                        print(f"[WARN] Failed to remove placeholder {placeholder}: {cleanup_err}")
                    job.placeholder_image = None

        if exit_code != 0:
            return exit_code

        final_dir = aggregate_outputs(jobs, aggregated_root, master_prompts)
        final_output_path = None
        if final_dir is not None:
            source_mode_dir = aggregated_root / args.mode.lower()

            if source_mode_dir.exists():
                # Merge into target directory without deleting existing runs
                target_mode_dir.mkdir(parents=True, exist_ok=True)

                # Move each child (e.g., PARAM_TAG) under the mode folder
                for child in sorted(source_mode_dir.iterdir()):
                    dest = target_mode_dir / child.name
                    if dest.exists():
                        # Avoid overwrite: append timestamp label
                        dest = target_mode_dir / f"{child.name}_{timestamp_label}"
                    shutil.move(str(child), str(dest))

                # Move metadata with a timestamped prefix to avoid clobbering
                for meta_name in ("prompts.txt", "merged_manifest.json"):
                    meta_src = aggregated_root / meta_name
                    if meta_src.exists():
                        stamped = target_mode_dir / f"{timestamp_label}_{meta_name}"
                        shutil.move(str(meta_src), str(stamped))

                try:
                    rel_path = Path(final_dir).relative_to(source_mode_dir)
                    final_output_path = target_mode_dir / rel_path
                except ValueError:
                    final_output_path = target_mode_dir
            else:
                # Fallback: move entire aggregated root under mode dir name
                target_mode_dir.mkdir(parents=True, exist_ok=True)
                for child in sorted(aggregated_root.iterdir()):
                    if child.name in {"prompts.txt", "merged_manifest.json"}:
                        # Handle metadata separately below
                        continue
                    dest = target_mode_dir / child.name
                    if dest.exists():
                        dest = target_mode_dir / f"{child.name}_{timestamp_label}"
                    shutil.move(str(child), str(dest))

                for meta_name in ("prompts.txt", "merged_manifest.json"):
                    meta_src = aggregated_root / meta_name
                    if meta_src.exists():
                        stamped = target_mode_dir / f"{timestamp_label}_{meta_name}"
                        shutil.move(str(meta_src), str(stamped))
                final_output_path = target_mode_dir

            print(f"[MERGE] Aggregated outputs merged into: {final_output_path}")
        else:
            print("[MERGE] No aggregated outputs were produced.")

        success = final_output_path is not None

        for job in jobs:
            try:
                if job.output_dir.exists():
                    shutil.rmtree(job.output_dir, ignore_errors=True)
                    print(f"[CLEANUP] Removed shard directory: {job.output_dir}")
            except Exception as cleanup_err:
                print(f"[WARN] Failed to remove shard directory {job.output_dir}: {cleanup_err}")
        return 0
    finally:
        if args.report_path:
            report_payload = {
                "success": success,
                "final_output_path": str(final_output_path) if final_output_path else None,
                "timestamp_label": timestamp_label,
                "target_mode_dir": str(target_mode_dir),
            }
            report_file = Path(args.report_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if not args.keep_temp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        if exit_code == 0:
            if aggregated_root.exists():
                shutil.rmtree(aggregated_root, ignore_errors=True)
            if stage_root.exists():
                shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
