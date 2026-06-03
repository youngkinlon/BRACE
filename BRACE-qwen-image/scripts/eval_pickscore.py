import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

PERSIST_ROOT = "/root/autodl-tmp"
PICKSCORE_LOCAL_DIR = f"{PERSIST_ROOT}/models/PickScore_v1"

os.environ.setdefault("HF_HOME", f"{PERSIST_ROOT}/cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_args():
    p = argparse.ArgumentParser(description="Single image PickScore evaluation")
    p.add_argument("--image", type=str, required=True, help="图片路径")
    p.add_argument("--prompt", type=str, required=True, help="对应的 prompt")
    p.add_argument("--model-dir", type=str, default=PICKSCORE_LOCAL_DIR)
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = args.model_dir
    if not Path(model_path, "config.json").exists():
        raise FileNotFoundError(f"PickScore model not found at {model_path}")

    print(f"Loading PickScore from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")

    image_inputs = processor(images=image, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    text_inputs = processor(text=args.prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

    with torch.no_grad():
        img_feat = model.get_image_features(**image_inputs)
        txt_feat = model.get_text_features(**text_inputs)

        if hasattr(img_feat, "pooler_output"):
            img_feat = img_feat.pooler_output
        if hasattr(txt_feat, "pooler_output"):
            txt_feat = txt_feat.pooler_output

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        score = model.logit_scale.exp() * (txt_feat @ img_feat.T)[0, 0]
        score = float(score.item())

    print(f"PickScore: {score:.4f}")


if __name__ == "__main__":
    main()
