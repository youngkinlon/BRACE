"""
批量测评前 N 个 prompt 的 ImageReward 分数。
用法:
    python eval_imagereward_batch.py --image_dir /path/to/images

图片目录中需包含 img_0.jpg ~ img_149.jpg（0-indexed，对应 txt 第1~150行）。
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import ImageReward as RM


def load_prompts(txt_path, num):
    """从 txt 文件加载前 num 条 prompt，每行一个 prompt。"""
    prompts = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > num:
                break
            prompt = line.strip()
            if prompt:
                prompts[i] = prompt
    return prompts


def find_image(image_dir, idx, offset=-1):
    """按指定 offset 查找图片。offset=-1 表示 txt 第1行严格对应 img_0.jpg。"""
    tidx = idx + offset

    candidates = [
        image_dir / f"img_{tidx}.jpg",
        image_dir / f"img_{tidx}.png",
        image_dir / f"img_{tidx:05d}.jpg",
        image_dir / f"img_{tidx:05d}.png",
        image_dir / f"{tidx}.jpg",
        image_dir / f"{tidx}.png",
        image_dir / f"{tidx:05d}.jpg",
        image_dir / f"{tidx:05d}.png",
    ]

    for p in candidates:
        if p.exists():
            return p
    return None

def parse_args():
    p = argparse.ArgumentParser(description="批量 ImageReward 测评前 N 条 prompt")
    p.add_argument("--image_dir", type=str, required=True, help="生成图片所在目录")
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    p.add_argument("--start", type=int, default=1, help="起始索引（1-indexed）")
    p.add_argument("--num", type=int, default=150, help="测评数量")
    p.add_argument("--output", type=str, default=None, help="结果保存路径（可选）")
    p.add_argument("--prompt_file", type=str, default="test-00000-of-00001.txt",
                   help="prompt 文本文件路径")
    p.add_argument("--img_offset", type=int, default=-1,
                   help="图片索引偏移量，-1 表示 txt 第1行对应 img_0.jpg，0 表示第1行对应 img_1.jpg")
    return p.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- 加载 prompt ----------
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompts = load_prompts(prompt_file, args.start + args.num - 1)
    prompts = {k: v for k, v in prompts.items() if k >= args.start}
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")
    if not prompts:
        print("  DEBUG: first 3 raw lines:")
        with open(prompt_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"    L{i+1}: {repr(line)}")
        return

    # ---------- 匹配图片 ----------
    image_paths = {}
    missing = []
    for idx in prompts:
        img_path = find_image(image_dir, idx, offset=args.img_offset)
        if img_path:
            image_paths[idx] = img_path
        else:
            missing.append(idx)

    if missing:
        print(f"WARNING: {len(missing)} images missing "
              f"(indices: {missing[:10]}{'...' if len(missing) > 10 else ''})")
    print(f"Found {len(image_paths)} images")
    if not image_paths:
        print("  DEBUG: listing image_dir contents:")
        for p in sorted(image_dir.iterdir())[:20]:
            print(f"    {p.name}")
        return

    # ---------- 加载模型 ----------
    print("Loading ImageReward model (ImageReward-v1.0)...")
    model = RM.load("ImageReward-v1.0", device=device)

    # ---------- 测评 ----------
    scores = []
    results = []

    for idx in tqdm(sorted(image_paths.keys()), desc="Evaluating ImageReward"):
        prompt = prompts[idx]
        img_path = image_paths[idx]
        if idx <= 3:
            print(f"\n[Debug] 配对检查 -> 图片: {img_path.name} | 提示词: '{prompt}'")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] idx={idx}: cannot open {img_path}: {e}")
            continue

        score = model.score(prompt, image)
        scores.append(score)
        results.append((idx, prompt, score))

    # ---------- 输出 ----------
    if not scores:
        print("No images evaluated.")
        return

    avg_score = sum(scores) / len(scores)
    print(f"\n===== ImageReward Results =====")
    print(f"Evaluated: {len(scores)} images")
    print(f"Average:   {avg_score:.4f}")
    print(f"Min:       {min(scores):.4f}")
    print(f"Max:       {max(scores):.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Average ImageReward: {avg_score:.4f}  (n={len(scores)})\n\n")
            for idx, prompt, score in results:
                f.write(f"{idx}\t{score:.4f}\t{prompt}\n")
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
