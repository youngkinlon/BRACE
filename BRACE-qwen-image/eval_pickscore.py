import argparse
import os
from pathlib import Path

# ============================================================
# Hugging Face / PickScore 缓存路径
# ============================================================
PERSIST_ROOT = "/root/autodl-tmp"

HF_CACHE_DIR = f"{PERSIST_ROOT}/cache/huggingface"
HF_HUB_CACHE_DIR = f"{HF_CACHE_DIR}/hub"
HF_XET_CACHE_DIR = f"{HF_CACHE_DIR}/xet"

PICKSCORE_REPO_ID = "yuvalkirstain/PickScore_v1"
PICKSCORE_LOCAL_DIR = f"{PERSIST_ROOT}/models/PickScore_v1"

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)
os.makedirs(HF_XET_CACHE_DIR, exist_ok=True)
os.makedirs(PICKSCORE_LOCAL_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE_DIR
os.environ["HF_XET_CACHE"] = HF_XET_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
from huggingface_hub import snapshot_download


def is_pickscore_downloaded(model_dir):
    model_dir = Path(model_dir)
    required_files = [
        "config.json", "model.safetensors", "preprocessor_config.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
    ]
    return all((model_dir / f).exists() for f in required_files)


def prepare_pickscore_model(repo_id=PICKSCORE_REPO_ID, local_dir=PICKSCORE_LOCAL_DIR):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if is_pickscore_downloaded(local_dir):
        print(f"Using local PickScore model: {local_dir}")
        return str(local_dir)

    print(f"Downloading PickScore from: {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), resume_download=True)

    if not is_pickscore_downloaded(local_dir):
        raise RuntimeError(f"PickScore download incomplete: {local_dir}")

    print(f"Download complete: {local_dir}")
    return str(local_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Single image PickScore evaluation")
    p.add_argument("--image", type=str, required=True, help="图片路径")
    p.add_argument("--prompt", type=str, required=True, help="对应的 prompt")
    p.add_argument("--model-dir", type=str, default=PICKSCORE_LOCAL_DIR)
    p.add_argument("--repo-id", type=str, default=PICKSCORE_REPO_ID)
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = prepare_pickscore_model(repo_id=args.repo_id, local_dir=args.model_dir)

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
