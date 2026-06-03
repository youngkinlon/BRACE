#!/usr/bin/env bash

# Chipmunk + FLUX 环境一键配置脚本（GPU 开发机用）
#
# 功能：
# - 创建独立的 conda 环境（chipmunk-flux）
# - 安装 PyTorch（需按本机实际通道调整）
# - 安装 Chipmunk 主仓库和 FLUX 示例代码
#
# 使用方式（在 GPU 开发机上）：
#   chmod +x CV/HiCache-Flux/models/chipmunk/set-up/setup_chipmunk_flux_env.sh
#   CV/HiCache-Flux/models/chipmunk/set-up/setup_chipmunk_flux_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHIPMUNK_DIR="${PROJECT_ROOT}/models/chipmunk"
ENV_NAME="chipmunk-flux"
ENV_NAME="${CHIPMUNK_ENV_NAME:-$ENV_NAME}"

echo "[STEP] 初始化 conda"
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] 未找到 conda 命令，请先安装 Miniconda/Anaconda 并确保 conda 在 PATH 中。"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[INFO] 检测到已有环境: ${ENV_NAME}，跳过创建步骤"
else
  echo "[STEP] 创建新的 conda 环境: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

echo "[STEP] 激活环境: ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "[STEP] 安装 PyTorch（请根据实际镜像/通道调整命令）"
echo "       当前示例使用 conda + CUDA 12.x，若已有合适的 torch，可手动跳过并注释本段。"

if python -c "import torch" >/dev/null 2>&1; then
  echo "[INFO] 当前环境已检测到 torch，跳过安装。"
else
  # 根据平台通道配置调整此命令（示例使用官方通道）
  conda install -y pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia || {
    echo "[WARN] 通过 conda 安装 PyTorch 失败，请根据本机通道配置手动安装合适版本的 torch>=2.5。"
  }
fi

echo "[STEP] 在当前环境中安装 Chipmunk 主仓库（编译 CUDA kernel）"
cd "${CHIPMUNK_DIR}"
pip install -e . --no-build-isolation

echo "[STEP] 安装 Chipmunk-FLUX 示例代码"
cd "${CHIPMUNK_DIR}/examples/flux"
pip install -e .

echo
echo "[DONE] Chipmunk + FLUX 环境安装完成。当前环境：${ENV_NAME}"
echo
echo "后续建议："
echo "1) 激活环境并运行示例："
echo "     source ${CONDA_BASE}/etc/profile.d/conda.sh"
echo "     conda activate ${ENV_NAME}"
echo "     cd ${CHIPMUNK_DIR}/examples/flux"
echo '     export PROMPT="A very cute cartoon chipmunk dressed up as a ninja holding katanas"'
echo "     python -m flux.cli --name flux-dev --prompt \"\$PROMPT\" --loop --chipmunk-config ./chipmunk-config.yml"
