#!/bin/bash
# ============================================================
# ChronoMagic-Bench UMT 评估脚本
# 同时计算 UMTScore 和 UMT-FVD 两个指标
# ============================================================
# 使用方式:
#   1. 填写下方 << 需要你填的参数 >> 区域的变量
#   2. bash run_eval.sh
# ============================================================

set -e

# ============================================================
# >>> 需要你填的参数 (共 4 个)
# ============================================================

# 写死: 视频路径 = $VIDEO_FOLDER/$MODEL_NAME/*.mp4
export MODELS=("chronomagic_outputs_taylor")
export VIDEO_FOLDER="/root/autodl-tmp/brace"
export TYPE="close"
export VERSION="150"

# ============================================================
# >>> 通常不需要改的参数
# ============================================================

# UMT 预训练权重 — 需从 HuggingFace 下载
# https://huggingface.co/laion/UnifiedMetric
export PRETRAINED="UMT-msrvtt-7k.pth"

# 如果连不上 HuggingFace:
# export HF_ENDPOINT=https://hf-mirror.com

# ============================================================
# 以下为执行逻辑，无需修改
# ============================================================

current_dir=$(pwd)
input_path_step3="results/UMTScore/${TYPE}/"
output_path_step3="results/UMTScore/${TYPE}/"
input_path_step4="results/UMTFVD/scores"
output_path_step4="results/UMTFVD/temp"

echo "=========================================="
echo "MODELS:       ${MODELS[@]}"
echo "VIDEO_FOLDER: $VIDEO_FOLDER"
echo "TYPE:         $TYPE"
echo "VERSION:      $VERSION"
echo "PRETRAINED:   $PRETRAINED"
echo "=========================================="

for model_name in "${MODELS[@]}"; do
    echo ""
    echo ">>> Evaluating model: $model_name"
    export MODEL_NAMES=$model_name

    echo "[Step 0] Extracting UMT-FVD features..."
    bash step0_get_umtfvd_feature.sh

    echo "[Step 1] Computing UMT-FVD scores..."
    bash step1_get_umtfvd.sh

    echo "[Step 2] Computing UMTScore..."
    bash step2_get_umtscope.sh

    echo "[Step 3] Merging UMTScore results..."
    python step3_get_merge_umt_scores.py --input_path "$input_path_step3" --output_path "$output_path_step3"

    echo "[Step 4] Merging UMT-FVD results..."
    python step4_get_merge_umt_fvd.py --input_path "$input_path_step4" --output_path "$output_path_step4"
done

echo ""
echo "Done!"
echo "UMTScore results:  results/UMTScore/${TYPE}/merge_umtscore_*.json"
echo "UMT-FVD results:   results/UMTFVD/scores/merge_umtfvd_${TYPE}.json"
