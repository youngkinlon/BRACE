"""
批量 PickScore 测评前 150 条 prompt。
用法:
    python eval_pickscore_150.py --image_dir /path/to/images
    python eval_pickscore_150.py --image_dir /path/to/images --prompt_file my_prompts.txt
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

PERSIST_ROOT = "/root/autodl-tmp"
PICKSCORE_LOCAL_DIR = f"{PERSIST_ROOT}/models/PickScore_v1"
os.environ.setdefault("HF_HOME", f"{PERSIST_ROOT}/cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_prompts(txt_path, start, num):
    """自动检测格式并加载 prompt。第1行以数字+tab开头=有行号；否则每行就是纯prompt。"""
    prompts = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        has_index = bool(first and "\t" in first and first.split("\t")[0].strip().isdigit())

        for i, line in enumerate(f, start=1):
            if i > start + num - 1:
                break
            line = line.strip()
            if not line:
                continue
            if has_index:
                parts = line.split("\t")
                prompt = parts[1].strip() if len(parts) >= 2 else ""
            else:
                prompt = line
            if prompt:
                prompts[i] = prompt
    return prompts


def find_images(image_dir, prompts, offset=-1):
    """匹配图片，返回 {idx: path}。"""
    image_dir = Path(image_dir)
    matched = {}
    for idx in prompts:
        found = None
        for delta in (0, offset):
            tidx = idx + delta
            for fmt in (f"img_{tidx}.jpg", f"img_{tidx}.png",
                        f"img_{tidx:05d}.jpg", f"img_{tidx:05d}.png",
                        f"{tidx}.jpg", f"{tidx}.png"):
                p = image_dir / fmt
                if p.exists():
                    found = p
                    break
            if found:
                break
        if found:
            matched[idx] = found
    return matched


def parse_args():
    p = argparse.ArgumentParser(description="批量 PickScore 测评前 150 条")
    p.add_argument("--image_dir", type=str, required=True, help="图片目录")
    p.add_argument("--num", type=int, default=150)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--img_offset", type=int, default=-1,
                   help="-1=第1行对应img_0, 0=第1行对应img_1")
    p.add_argument("--prompt_file", type=str, default="test-00000-of-00001.txt")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载 prompt
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"找不到文件: {prompt_file}  (cwd={os.getcwd()})")
    prompts = load_prompts(prompt_file, args.start, args.num)
    print(f"Loaded {len(prompts)} prompts")

    # 2. 匹配图片
    image_paths = find_images(args.image_dir, prompts, offset=args.img_offset)
    missing = [idx for idx in prompts if idx not in image_paths]
    print(f"Found {len(image_paths)} images, missing {len(missing)}")
    if not image_paths:
        print("图片目录内容:")
        for p in sorted(Path(args.image_dir).iterdir())[:30]:
            print(f"  {p.name}")
        return

    # 3. 加载模型
    model_path = PICKSCORE_LOCAL_DIR
    print(f"Loading PickScore: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()

    # 4. 测评
    scores, results = [], []
    for idx in tqdm(sorted(image_paths), desc="PickScore"):
        img = Image.open(image_paths[idx]).convert("RGB")
        img_in = processor(images=img, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        txt_in = processor(text=prompts[idx], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

        with torch.no_grad():
            ifeat = model.get_image_features(**img_in)
            tfeat = model.get_text_features(**txt_in)
            ifeat = ifeat.pooler_output if hasattr(ifeat, "pooler_output") else ifeat
            tfeat = tfeat.pooler_output if hasattr(tfeat, "pooler_output") else tfeat
            ifeat = F.normalize(ifeat, dim=-1)
            tfeat = F.normalize(tfeat, dim=-1)
            score = float((model.logit_scale.exp() * (tfeat @ ifeat.T)[0, 0]).item())

        scores.append(score)
        results.append((idx, prompts[idx], score))

    avg = sum(scores) / len(scores)
    print(f"\nAvg: {avg:.4f}  Min: {min(scores):.4f}  Max: {max(scores):.4f}  N: {len(scores)}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(f"Avg PickScore: {avg:.4f} (n={len(scores)})\n\n")
            for idx, prompt, score in results:
                f.write(f"{idx}\t{score:.4f}\t{prompt}\n")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
