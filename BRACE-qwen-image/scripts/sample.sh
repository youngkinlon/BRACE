#!/bin/bash

# 预处理：固定工作目录与venv环境
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/resources/weights"
cd "$PROJECT_ROOT" || { echo "[ERROR] 进入项目目录失败"; exit 1; }
echo "[INFO] 工作目录: $(pwd)"

## 虚拟环境处理：优先使用已激活环境；若未激活则尝试本地 .venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate || echo "[WARN] 激活 .venv 失败，将继续使用当前环境"
    [[ -n "${VIRTUAL_ENV:-}" ]] && echo "[INFO] 已激活虚拟环境: .venv"
  else
    echo "[WARN] 未检测到已激活的虚拟环境且本地 .venv 不存在，继续使用当前 Python 环境"
  fi
else
  echo "[INFO] 已激活虚拟环境: $(basename "$VIRTUAL_ENV")"
fi

# 0️⃣  统一临时与缓存目录，避免根分区占用
export TEMP_ROOT="${TEMP_ROOT:-$PROJECT_ROOT/.cache/tmp}"
export TMPDIR="${TMPDIR:-$TEMP_ROOT}"
export TMP="${TMP:-$TEMP_ROOT}"
export TEMP="${TEMP:-$TEMP_ROOT}"
export HF_HOME="${HF_HOME:-$TEMP_ROOT/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TEMP_ROOT/.huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TEMP_ROOT/.transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" || true

## 1️⃣  环境一致性：尽量使用当前仓库下的 .venv，但不做强制
EXPECTED_VENV="$PROJECT_ROOT/.venv"
if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]]; then
  echo "[WARN] 虚拟环境路径不匹配"
  echo "      当前: $VIRTUAL_ENV"
  echo "      期望: $EXPECTED_VENV"
  if [[ -f "$EXPECTED_VENV/bin/activate" ]]; then
    echo "[INFO] 尝试切换到本地 .venv ..."
    # shellcheck disable=SC1091
    source "$EXPECTED_VENV/bin/activate" || echo "[WARN] 切换失败，继续使用当前环境"
  fi
fi

echo "[INFO] 使用 Python: $(command -v python)"

# 默认值
MODE="Taylor" # Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa
MODEL_NAME="flux-dev" # flux-dev |  
INTERVAL="5"
MAX_ORDER="2"
OUTPUT_DIR="$PROJECT_ROOT/results/flux"
PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt.txt"
WIDTH=1024
HEIGHT=1024
NUM_STEPS=50
NUM_STEPS_SET=false
LIMIT=10
HICACHE_SCALE_FACTOR="0.7"
ANALYTIC_SIGMA_ALPHA=""
ANALYTIC_SIGMA_MAX=""
ANALYTIC_SIGMA_BETA=""
ANALYTIC_SIGMA_EPS=""
ANALYTIC_SIGMA_Q_QUANTILE=""
ANALYTIC_SIGMA_SMOOTH=""
START_INDEX=0
MODEL_DIR=""
FIRST_ENHANCE=3

# --- ClusCa 默认参数 ----------------------------------------------------------
CLUSCA_FRESH_THRESHOLD=5        # ClusCa fresh 阈值
CLUSCA_CLUSTER_NUM=16           # 聚类数量
CLUSCA_CLUSTER_METHOD="kmeans"  # 聚类方法 (kmeans/kmeans++/random)
CLUSCA_K=1                      # 每个聚类选择的 fresh token 数
CLUSCA_PROPAGATION_RATIO=0.005  # 特征传播比例
# ------------------------------------------------------------------------------

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -m, --mode MODE           缓存模式 (Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa) [默认: Taylor]"
    echo "  --model_name NAME         FLUX 模型 (flux-dev|flux-schnell) [默认: flux-dev]"
    echo "  --model_dir DIR          指定本地 FLUX 权重目录(包含 flow 与 ae)"
    echo "  -i, --interval INTERVAL   间隔值 [默认: 1]"
    echo "  -o, --max_order ORDER     最大阶数 [默认: 1]"
    echo "  -d, --output_dir DIR      输出目录 [默认: $PROJECT_ROOT/results/flux]"
echo "  -p, --prompt_file FILE    提示文件 [默认: resources/prompts/prompt.txt]"
    echo "  -w, --width WIDTH         图像宽度 [默认: 1024]"
    echo "  -h, --height HEIGHT       图像高度 [默认: 1024]"
    echo "  -s, --num_steps STEPS     采样步数 [默认: 50]"
    echo "  -l, --limit LIMIT         测试数量限制 [默认: 10]"
    echo "  --hicache_scale FACTOR    HiCache多项式缩放因子 [默认: 0.5]"
    echo "  --first_enhance N         初始增强步数 (前 N 步强制 full) [默认: 3]"
    echo "  --start_index N            结果文件编号偏移量 [默认: 0]"
    echo "  --clusca_fresh_threshold N  ClusCa: fresh 阈值 [默认: 5]"
    echo "  --clusca_cluster_num N    ClusCa: 聚类数量 [默认: 16]"
    echo "  --clusca_cluster_method M ClusCa: 聚类方法 (kmeans/kmeans++/random) [默认: kmeans]"
    echo "  --clusca_k N              ClusCa: 每个聚类选择 fresh token 数 [默认: 1]"
    echo "  --clusca_propagation_ratio R  ClusCa: 特征传播比例 [默认: 0.005]"
    echo "  --analytic_sigma_alpha VAL    HiCache-Analytic: 解析 sigma 的 alpha（默认 1.28）"
    echo "  --analytic_sigma_max VAL      HiCache-Analytic: 解析 sigma 的上限（默认 1.0）"
    echo "  --analytic_sigma_beta VAL     HiCache-Analytic: q 的 EMA 系数（默认 0.01，<=0 关闭在线更新）"
    echo "  --analytic_sigma_eps VAL      HiCache-Analytic: 解析公式分母的 epsilon（默认 1e-6）"
    echo "  --analytic_sigma_q_quantile Q HiCache-Analytic: q 的分位数估计（如 0.95；缺省则用均值）"
    echo "  --analytic_sigma_smooth VAL   HiCache-Analytic: σ 的 log 域平滑系数 gamma（0 表示不平滑）"
    echo "  --help                    显示帮助信息"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -o|--max_order)
            MAX_ORDER="$2"
            shift 2
            ;;
        -d|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--prompt_file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
            shift 2
            ;;
        -s|--num_steps)
            NUM_STEPS="$2"
            NUM_STEPS_SET=true
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --hicache_scale)
            HICACHE_SCALE_FACTOR="$2"
            shift 2
            ;;
        --first_enhance)
            FIRST_ENHANCE="$2"
            shift 2
            ;;
        --start_index)
            START_INDEX="$2"
            shift 2
            ;;
        --clusca_fresh_threshold)
            CLUSCA_FRESH_THRESHOLD="$2"
            shift 2
            ;;
        --clusca_cluster_num)
            CLUSCA_CLUSTER_NUM="$2"
            shift 2
            ;;
        --clusca_cluster_method)
            CLUSCA_CLUSTER_METHOD="$2"
            shift 2
            ;;
        --clusca_k)
            CLUSCA_K="$2"
            shift 2
            ;;
        --clusca_propagation_ratio)
            CLUSCA_PROPAGATION_RATIO="$2"
            shift 2
            ;;
        --analytic_sigma_alpha)
            ANALYTIC_SIGMA_ALPHA="$2"
            shift 2
            ;;
        --analytic_sigma_max)
            ANALYTIC_SIGMA_MAX="$2"
            shift 2
            ;;
        --analytic_sigma_beta)
            ANALYTIC_SIGMA_BETA="$2"
            shift 2
            ;;
        --analytic_sigma_eps)
            ANALYTIC_SIGMA_EPS="$2"
            shift 2
            ;;
        --analytic_sigma_q_quantile)
            ANALYTIC_SIGMA_Q_QUANTILE="$2"
            shift 2
            ;;
        --analytic_sigma_smooth)
            ANALYTIC_SIGMA_SMOOTH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证模式
if [[ "$MODE" != "Taylor" && "$MODE" != "Taylor-Scaled" && "$MODE" != "HiCache" && "$MODE" != "HiCache-Analytic" && "$MODE" != "original" && "$MODE" != "ToCa" && "$MODE" != "Delta" && "$MODE" != "collect" && "$MODE" != "ClusCa" && "$MODE" != "Hi-ClusCa" ]]; then
    echo "错误: 不支持的模式 '$MODE'"
    echo "支持的模式: Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa"
    exit 1
fi

# 设置环境变量
# 模型路径优先使用本地缓存，找不到时回退到远端权重名称
export T5_DIR="$WEIGHTS_DIR/t5-v1_1-xxl"

resolve_clip_local() {
    local base_candidates=(
        "$WEIGHTS_DIR/clip-vit-large-patch14"
        "$WEIGHTS_DIR/clip-vit-large-patch14/clip-vit-large-patch14"
        "$WEIGHTS_DIR/openai/clip-vit-large-patch14"
    )

    for candidate in "${base_candidates[@]}"; do
        if [[ -d "$candidate" && -f "$candidate/config.json" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    local hub_root="${HF_HOME:-$TEMP_ROOT/.hf_home}/hub/models--openai--clip-vit-large-patch14/snapshots"
    if [[ -d "$hub_root" ]]; then
        local latest
        latest=$(ls -1dt "$hub_root"/* 2>/dev/null | head -n1 || true)
        if [[ -n "$latest" && -f "$latest/config.json" ]]; then
            echo "$latest"
            return 0
        fi
    fi

    local found
    found=$(find "$WEIGHTS_DIR" -type f -name config.json -path "*clip-vit-large-patch14*" 2>/dev/null | head -n1 || true)
    if [[ -n "$found" ]]; then
        echo "$(dirname "$found")"
        return 0
    fi

    return 1
}

CLIP_LOCAL_DIR="$(resolve_clip_local || true)"
if [[ -n "$CLIP_LOCAL_DIR" ]]; then
    export CLIP_DIR="$CLIP_LOCAL_DIR"
    export HF_HUB_OFFLINE="1"
    export HF_DATASETS_OFFLINE="1"
    export TRANSFORMERS_OFFLINE="1"
    echo "[INFO] 使用本地 CLIP 模型: $CLIP_DIR"
else
    export CLIP_DIR="openai/clip-vit-large-patch14"
    export HF_HUB_OFFLINE="0"
    export HF_DATASETS_OFFLINE="0"
    export TRANSFORMERS_OFFLINE="0"
    echo "[WARN] 未发现本地 CLIP 缓存，将尝试联网下载 openai/clip-vit-large-patch14"
fi

# 根据模型名称自动匹配模型目录路径
auto_detect_model_dir() {
    local model_name="$1"
    local candidates=()
    
    if [[ "$model_name" == "flux-schnell" ]]; then
        candidates=(
            "$WEIGHTS_DIR/FLUX.1-schnell"
            "$WEIGHTS_DIR/flux.schnell"
            "$WEIGHTS_DIR/flux-schnell"
            "$WEIGHTS_DIR/schnell"
        )
    else
        candidates=(
            "$WEIGHTS_DIR/FLUX.1-dev"
            "$WEIGHTS_DIR/flux.dev"
            "$WEIGHTS_DIR/flux-dev"
            "$WEIGHTS_DIR/dev"
        )
    fi
    
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    
    return 1
}

# 设置模型目录
if [[ -n "$MODEL_DIR" ]]; then
    echo "[INFO] 指定了 --model_dir: $MODEL_DIR"
    AUTO_MODEL_DIR="$MODEL_DIR"
else
    AUTO_MODEL_DIR="$(auto_detect_model_dir "$MODEL_NAME")"
    if [[ -z "$AUTO_MODEL_DIR" ]]; then
        echo "[ERROR] 未找到匹配的模型目录，请检查 resources/weights/ 目录或使用 --model_dir 指定"
        echo "支持的目录格式:"
        if [[ "$MODEL_NAME" == "flux-schnell" ]]; then
            echo "  - resources/weights/FLUX.1-schnell"
            echo "  - resources/weights/flux.schnell"
            echo "  - resources/weights/flux-schnell"
            echo "  - resources/weights/schnell"
        else
            echo "  - resources/weights/FLUX.1-dev"
            echo "  - resources/weights/flux.dev"
            echo "  - resources/weights/flux-dev"
            echo "  - resources/weights/dev"
        fi
        exit 1
    else
        echo "[INFO] 自动检测到模型目录: $AUTO_MODEL_DIR"
    fi
fi

# 根据模型名称设置权重路径
if [[ "$MODEL_NAME" == "flux-schnell" ]]; then
    if [[ -f "$AUTO_MODEL_DIR/flux1-schnell.safetensors" ]]; then
        export FLUX_SCHNELL="$AUTO_MODEL_DIR/flux1-schnell.safetensors"
        echo "[INFO] 使用 FLUX_SCHNELL: $FLUX_SCHNELL"
    else
        echo "[ERROR] 未在 $AUTO_MODEL_DIR 找到 flux1-schnell.safetensors"
        exit 1
    fi
else
    if [[ -f "$AUTO_MODEL_DIR/flux1-dev.safetensors" ]]; then
        export FLUX_DEV="$AUTO_MODEL_DIR/flux1-dev.safetensors"
        echo "[INFO] 使用 FLUX_DEV: $FLUX_DEV"
    else
        echo "[ERROR] 未在 $AUTO_MODEL_DIR 找到 flux1-dev.safetensors"
        exit 1
    fi
fi

if [[ -f "$AUTO_MODEL_DIR/ae.safetensors" ]]; then
    export AE="$AUTO_MODEL_DIR/ae.safetensors"
    echo "[INFO] 使用 AE: $AE"
else
    echo "[ERROR] 未在 $AUTO_MODEL_DIR 找到 ae.safetensors"
    exit 1
fi

# 在确定目录前，若未显式指定步数且为 schnell，设置默认步数为 4
if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    NUM_STEPS=4
fi

# 统一的输出目录：仅保留一级目录为 mode，其余关键参数合并为子目录名
MODE_LOWER="${MODE,,}"
if [[ "$MODE" == "HiCache-Analytic" ]]; then
    # HiCache-Analytic 不再使用 hicache_scale 作为语义参数，避免与解析 σ 混淆
    PARAM_TAG="mn_${MODEL_NAME}_i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_analytic"
else
    PARAM_TAG="mn_${MODEL_NAME}_i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_hs_${HICACHE_SCALE_FACTOR}"
fi
FULL_OUTPUT_DIR="$OUTPUT_DIR/${MODE_LOWER}/${PARAM_TAG}"
mkdir -p "$FULL_OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "$FULL_OUTPUT_DIR" > "$OUTPUT_DIR/.full_output_dir"

# 创建限制数量的临时prompt文件
TEMP_PROMPT_FILE="$FULL_OUTPUT_DIR/temp_prompts.txt"
head -n "$LIMIT" "$PROMPT_FILE" > "$TEMP_PROMPT_FILE"

# 显示配置信息
echo "================================="
echo "图像生成配置:"
echo "模式: $MODE"
echo "间隔: $INTERVAL"
echo "最大阶数: $MAX_ORDER"
echo "输出目录: $FULL_OUTPUT_DIR"
echo "提示文件: $PROMPT_FILE"
echo "FLUX 模型: $MODEL_NAME"
if [[ -n "$MODEL_DIR" ]]; then
    echo "模型目录: $MODEL_DIR"
elif [[ -n "$AUTO_MODEL_DIR" ]]; then
    echo "自动检测模型目录: $AUTO_MODEL_DIR"
fi
if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    # 若用户未显式指定步数，schnell 默认步数为 4
    NUM_STEPS=4
fi

echo "图像尺寸: ${WIDTH}x${HEIGHT}"
echo "采样步数: $NUM_STEPS"
echo "测试数量限制: $LIMIT"
echo "HiCache缩放因子: $HICACHE_SCALE_FACTOR"
echo "First enhance: $FIRST_ENHANCE"
echo "起始索引: $START_INDEX"
if [[ "$MODE" == "ClusCa" ]]; then
    echo "ClusCa fresh 阈值: $CLUSCA_FRESH_THRESHOLD"
    echo "ClusCa 聚类: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "ClusCa k: $CLUSCA_K, 传播比例: $CLUSCA_PROPAGATION_RATIO"
elif [[ "$MODE" == "Hi-ClusCa" ]]; then
    echo "Hi-ClusCa fresh 阈值: $CLUSCA_FRESH_THRESHOLD"
    echo "Hi-ClusCa 聚类: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "Hi-ClusCa k: $CLUSCA_K, 传播比例: $CLUSCA_PROPAGATION_RATIO"
    echo "Hi-ClusCa HiCache 缩放: $HICACHE_SCALE_FACTOR"
fi
echo "================================="

CLUSCA_ARGS=()
if [[ "$MODE" == "ClusCa" || "$MODE" == "Hi-ClusCa" ]]; then
    CLUSCA_ARGS+=(--clusca_fresh_threshold "$CLUSCA_FRESH_THRESHOLD")
    CLUSCA_ARGS+=(--clusca_cluster_num "$CLUSCA_CLUSTER_NUM")
    CLUSCA_ARGS+=(--clusca_cluster_method "$CLUSCA_CLUSTER_METHOD")
    CLUSCA_ARGS+=(--clusca_k "$CLUSCA_K")
    CLUSCA_ARGS+=(--clusca_propagation_ratio "$CLUSCA_PROPAGATION_RATIO")
fi

ANALYTIC_ARGS=()
if [[ "$MODE" == "HiCache-Analytic" ]]; then
    if [[ -n "$ANALYTIC_SIGMA_ALPHA" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_alpha "$ANALYTIC_SIGMA_ALPHA")
    fi
    if [[ -n "$ANALYTIC_SIGMA_MAX" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_max "$ANALYTIC_SIGMA_MAX")
    fi
    if [[ -n "$ANALYTIC_SIGMA_BETA" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_beta "$ANALYTIC_SIGMA_BETA")
    fi
    if [[ -n "$ANALYTIC_SIGMA_EPS" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_eps "$ANALYTIC_SIGMA_EPS")
    fi
    if [[ -n "$ANALYTIC_SIGMA_Q_QUANTILE" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_q_quantile "$ANALYTIC_SIGMA_Q_QUANTILE")
    fi
    if [[ -n "$ANALYTIC_SIGMA_SMOOTH" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_smooth "$ANALYTIC_SIGMA_SMOOTH")
    fi
fi

# 执行采样
echo "开始生成图像..."
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python "$PROJECT_ROOT/models/flux/src/sample.py" \
  --prompt_file "$TEMP_PROMPT_FILE" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --model_name "$MODEL_NAME" \
  --add_sampling_metadata \
  --output_dir "$FULL_OUTPUT_DIR" \
  --num_steps "$NUM_STEPS" \
  --cache_mode "$MODE" \
  --interval "$INTERVAL" \
  --max_order "$MAX_ORDER" \
  --first_enhance "$FIRST_ENHANCE" \
  --seed 0 \
  --start_index "$START_INDEX" \
  --hicache_scale "$HICACHE_SCALE_FACTOR" \
  "${CLUSCA_ARGS[@]}" \
  "${ANALYTIC_ARGS[@]}"

PYTHON_EXIT_CODE=$?
if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] 图像生成脚本执行失败 (退出码: $PYTHON_EXIT_CODE)"
    rm -f "$TEMP_PROMPT_FILE"
    exit $PYTHON_EXIT_CODE
fi

echo "图像生成完成！"
echo "输出目录: $FULL_OUTPUT_DIR"

# 清理临时文件
rm -f "$TEMP_PROMPT_FILE"

echo ""
# 若位于多卡临时目录，避免打印误导性评估命令（聚合器将给出最终建议）
case "$OUTPUT_DIR" in
  *"/.multi_gpu_tmp/"*)
    :
    ;;
  *)
    echo "================================="
    # 固定 GT 建议目录为 Taylor baseline interval_1/order_2（单卡场景）
    GT_SUGGEST="$PROJECT_ROOT/results/taylor/interval_1/order_2"
    echo "建议执行评估命令:"
    echo "  bash \"$PROJECT_ROOT/evaluation/run_eval.sh\" --acc \"$FULL_OUTPUT_DIR\" --gt \"$GT_SUGGEST\""
    echo "================================="
    ;;
esac
