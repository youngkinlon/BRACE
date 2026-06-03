#!/usr/bin/env python3
"""从 ModelScope 下载 Qwen-Image 完整模型，适配 HiCache 的 from_pretrained 加载。"""
import sys
from modelscope import snapshot_download

LOCAL_DIR = sys.argv[1] if len(sys.argv) > 1 else "./premodels/Qwen/Qwen-Image"

print(f"[INFO] 从 ModelScope 下载 Qwen/Qwen-Image 到 {LOCAL_DIR} ...")
snapshot_download(
    "Qwen/Qwen-Image",
    cache_dir=LOCAL_DIR,
)

print(f"[DONE] 模型已下载到 {LOCAL_DIR}")
