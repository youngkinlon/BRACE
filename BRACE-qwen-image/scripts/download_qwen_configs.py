#!/usr/bin/env python3
"""下载 Qwen-Image 缺失的配置文件（config.json 等），不重复下载已存在的权重。"""
import os
import sys
from huggingface_hub import hf_hub_download

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN_IMAGE_MODEL_PATH", "")
REPO = "Qwen/Qwen-Image"

FILES = [
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "vae/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "scheduler/scheduler_config.json",
]

if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
    print(f"Usage: python {__file__} /path/to/Qwen-Image")
    sys.exit(1)

for f in FILES:
    local = os.path.join(MODEL_DIR, f)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if os.path.exists(local):
        print(f"[SKIP] {f}")
    else:
        print(f"[DOWNLOAD] {f} ...")
        hf_hub_download(repo_id=REPO, filename=f, local_dir=MODEL_DIR)

print("Done.")