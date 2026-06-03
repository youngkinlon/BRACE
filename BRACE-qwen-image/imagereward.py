import argparse
import torch
from PIL import Image
import ImageReward as RM


def main():
    parser = argparse.ArgumentParser(description="单张图片 ImageReward 评分")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument("--prompt", type=str, required=True, help="对应的 prompt")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ImageReward model...")
    model = RM.load("ImageReward-v1.0", device=device)

    image = Image.open(args.image).convert("RGB")
    score = model.score(args.prompt, image)

    print("=" * 80)
    print(f"Image:        {args.image}")
    print(f"Prompt:       {args.prompt}")
    print(f"ImageReward:  {score:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
