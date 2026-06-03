#!/bin/bash

# 预处理：固定工作目录与 venv 环境
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "[ERROR] 进入项目目录失败"; exit 1; }
echo "[INFO] 工作目录: $(pwd)"

# 统一临时与缓存目录，避免根分区占用
export TEMP_ROOT="${TEMP_ROOT:-$PROJECT_ROOT/.cache/tmp}"
export TMPDIR="${TMPDIR:-$TEMP_ROOT}"
export TMP="${TMP:-$TEMP_ROOT}"
export TEMP="${TEMP:-$TEMP_ROOT}"
export HF_HOME="${HF_HOME:-$TEMP_ROOT/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TEMP_ROOT/.huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TEMP_ROOT/.transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" || true

BACKEND="flux"
PYTHON_PATH=""

# 默认配置
MODE="Taylor"  # Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa
MODE_SET=false
MODEL_NAME="flux-dev"  # flux-dev | flux-schnell
INTERVAL="5"
MAX_ORDER="2"
WIDTH=1024
HEIGHT=1024
WIDTH_SET=false
HEIGHT_SET=false
NUM_STEPS=50
NUM_STEPS_SET=false
LIMIT=200
GUIDANCE=3.5
HICACHE_SCALE_FACTOR="0.6"
FIRST_ENHANCE="3"
PROMPT_FILE_DEFAULT_FLUX="$PROJECT_ROOT/resources/prompts/prompt.txt"
PROMPT_FILE_DEFAULT_QWEN="$PROJECT_ROOT/models/qwen_image/prompts/DrawBench200.txt"
PROMPT_FILE_DEFAULT_CHIPMUNK="$PROJECT_ROOT/resources/prompts/prompt.txt"
PROMPT_FILE="$PROMPT_FILE_DEFAULT_FLUX"
PROMPT_FILE_SET=false
BASE_OUTPUT_DIR="$PROJECT_ROOT/results"
BASE_OUTPUT_DIR_SET=false
WEIGHTS_DIR="$PROJECT_ROOT/resources/weights"
GPU_LIST=""
NUM_GPUS=""
RUN_NAME=""
AUTO_RUN_NAME=false
KEEP_TEMP=false
DRY_RUN=false
MODEL_DIR=""
QWEN_MODEL_PATH=""
FORCE=false
# Qwen-Image 环境默认配置（可通过环境变量覆盖）
DEFAULT_QWEN_ACTIVATE=""
DEFAULT_QWEN_ENV=""
QWEN_ACTIVATE_SCRIPT="${QWEN_IMAGE_ACTIVATE_SCRIPT:-$DEFAULT_QWEN_ACTIVATE}"
QWEN_ENV_PATH="${QWEN_IMAGE_ENV_PATH:-$DEFAULT_QWEN_ENV}"
QWEN_ENV_OVERRIDE=""
CHIPMUNK_CONFIG="$PROJECT_ROOT/models/chipmunk/examples/flux/chipmunk-config.yml"

# ClusCa 默认参数
CLUSCA_FRESH_THRESHOLD=5
CLUSCA_CLUSTER_NUM=16
CLUSCA_CLUSTER_METHOD="kmeans"
CLUSCA_K=1
CLUSCA_PROPAGATION_RATIO=0.005

EXTRA_SAMPLE_ARGS=()

show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "      --backend BACKEND       运行后端 (flux|qwen-image|chipmunk) [默认: flux]"
    echo "  -m, --mode MODE             缓存模式 (Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa)"
    echo "      --model_name NAME       FLUX 模型 (flux-dev|flux-schnell) [默认: flux-dev]"
    echo "  -i, --interval N            采样间隔 [默认: 5]"
    echo "  -o, --max_order N           泰勒最大阶数 [默认: 2]"
    echo "      --first_enhance N        初始增强步数 (前 N 步强制 full) [默认: 3]"
    echo "  -d, --output_dir DIR        结果基础目录 (多卡运行会在其下创建子目录)"
    echo "  -p, --prompt_file FILE      Prompt 列表 [默认: resources/prompts/prompt.txt]"
    echo "  -w, --width WIDTH           图像宽度 [默认: 1024]"
    echo "  -h, --height HEIGHT         图像高度 [默认: 1024]"
    echo "  -s, --num_steps STEPS       采样步数 [默认: 50]"
    echo "  -l, --limit LIMIT           Prompt 限制数量 [默认: 10]"
    echo "      --guidance VALUE        Chipmunk-Flux: guidance 参数 [默认: 3.5]"
    echo "      --python PATH           指定运行 multi_gpu_launcher.py 的 Python 解释器"
    echo "  --gpus IDS                  指定 GPU 列表 (示例: 0,1,3)"
    echo "  --num_gpus N                未指定 --gpus 时自动从 0 开始取 N 张卡"
    echo "  --run-name NAME             自定义运行名 (用于输出目录)"
    echo "  --hicache_scale FACTOR      HiCache 多项式缩放因子 [默认: 0.7]"
    echo "  --model_dir DIR             指定本地 FLUX 权重目录(包含 flow 与 ae)"
    echo "      --model_path PATH       Qwen-Image: 指定模型 checkpoint 路径"
    echo "      --qwen_env PATH         Qwen-Image: 覆盖默认 Conda 环境路径"
    echo "  --fresh_threshold VALUE     ClusCa: fresh 阈值 [默认: 5]"
    echo "  --cluster_num N             ClusCa: 聚类数量 [默认: 16]"
    echo "  --cluster_method NAME       ClusCa: 聚类方法 [默认: kmeans]"
    echo "  --k N                       ClusCa: 每个聚类选择 fresh token 数 [默认: 1]"
    echo "  --propagation_ratio VALUE   ClusCa: 特征传播比例 [默认: 0.005]"
    echo "  --keep-temp                 保留 RUN/ 下的 Prompt 切分文件"
    echo "      --chipmunk_config PATH  Chipmunk-Flux: 配置文件路径 [默认: models/chipmunk/examples/flux/chipmunk-config.yml]"
    echo "  --dry-run                   仅打印即将执行的命令"
    echo "  --help                      显示帮助信息"
    echo "  --                          之后的参数原样透传给 sample.sh"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            MODE_SET=true
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
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
        --first_enhance)
            FIRST_ENHANCE="$2"
            shift 2
            ;;
        -d|--output_dir)
            BASE_OUTPUT_DIR="$2"
            BASE_OUTPUT_DIR_SET=true
            shift 2
            ;;
        -p|--prompt_file)
            PROMPT_FILE="$2"
            PROMPT_FILE_SET=true
            shift 2
            ;;
        -w|--width)
            WIDTH="$2"
            WIDTH_SET=true
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
            HEIGHT_SET=true
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
        --guidance)
            GUIDANCE="$2"
            shift 2
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --num_gpus|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --run-name|--run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_PATH="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --model_path)
            QWEN_MODEL_PATH="$2"
            shift 2
            ;;
        --qwen_env)
            QWEN_ENV_OVERRIDE="$2"
            shift 2
            ;;
        --hicache_scale)
            HICACHE_SCALE_FACTOR="$2"
            shift 2
            ;;
        --fresh_threshold)
            CLUSCA_FRESH_THRESHOLD="$2"
            shift 2
            ;;
        --cluster_num)
            CLUSCA_CLUSTER_NUM="$2"
            shift 2
            ;;
        --cluster_method)
            CLUSCA_CLUSTER_METHOD="$2"
            shift 2
            ;;
        --k)
            CLUSCA_K="$2"
            shift 2
            ;;
        --propagation_ratio)
            CLUSCA_PROPAGATION_RATIO="$2"
            shift 2
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --chipmunk_config)
            CHIPMUNK_CONFIG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        --)
            shift
            EXTRA_SAMPLE_ARGS+=("$@")
            break
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ "$BACKEND" != "flux" && "$BACKEND" != "qwen-image" && "$BACKEND" != "chipmunk" ]]; then
    echo "[ERROR] 不支持的 backend: $BACKEND (仅支持 flux、qwen-image 或 chipmunk)"
    exit 1
fi

if [[ "$BACKEND" == "chipmunk" && "$MODE_SET" != true ]]; then
    MODE="chipmunk"
fi

if [[ "$PROMPT_FILE_SET" != true ]]; then
    if [[ "$BACKEND" == "qwen-image" ]]; then
        PROMPT_FILE="$PROMPT_FILE_DEFAULT_QWEN"
    elif [[ "$BACKEND" == "chipmunk" ]]; then
        PROMPT_FILE="$PROMPT_FILE_DEFAULT_CHIPMUNK"
    else
        PROMPT_FILE="$PROMPT_FILE_DEFAULT_FLUX"
    fi
fi

if [[ "$BASE_OUTPUT_DIR_SET" != true ]]; then
    case "$BACKEND" in
        flux)
            BASE_OUTPUT_DIR="$PROJECT_ROOT/results/flux"
            ;;
        chipmunk)
            BASE_OUTPUT_DIR="$PROJECT_ROOT/results/chipmunk"
            ;;
        qwen-image)
            BASE_OUTPUT_DIR="$PROJECT_ROOT/results/qwen-image"
            ;;
    esac
fi

if [[ "$BACKEND" == "qwen-image" ]]; then
    if [[ "$WIDTH_SET" != true ]]; then
        WIDTH=1328
    fi
    if [[ "$HEIGHT_SET" != true ]]; then
        HEIGHT=1328
    fi
fi

if [[ "$BACKEND" == "flux" ]]; then
    if [[ "$MODE" != "Taylor" && "$MODE" != "Taylor-Scaled" && "$MODE" != "HiCache" && "$MODE" != "HiCache-Analytic" && "$MODE" != "original" && "$MODE" != "ToCa" && "$MODE" != "Delta" && "$MODE" != "collect" && "$MODE" != "ClusCa" && "$MODE" != "Hi-ClusCa" ]]; then
        echo "错误: 不支持的模式 '$MODE'"
        echo "支持的模式: Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa"
        exit 1
    fi
elif [[ "$BACKEND" == "chipmunk" ]]; then
    mode_lower="${MODE,,}"
    case "$mode_lower" in
        chipmunk|original|taylor|hicache)
            ;;
        *)
            echo "错误: backend 'chipmunk' 不支持模式 '$MODE'"
            echo "支持的模式: chipmunk, original, Taylor, HiCache"
            exit 1
            ;;
    esac
else
    case "$MODE" in
        Taylor|HiCache|original|ToCa|Delta|GroupedTaylor)
            ;;
        *)
            echo "错误: backend 'qwen-image' 不支持模式 '$MODE'"
            echo "支持的模式: Taylor, HiCache, original, ToCa, Delta, GroupedTaylor"
            exit 1
            ;;
    esac
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[ERROR] Prompt 文件不存在: $PROMPT_FILE"
    exit 1
fi

if [[ "$BACKEND" == "chipmunk" && ! -f "$CHIPMUNK_CONFIG" ]]; then
    echo "[ERROR] Chipmunk 配置文件不存在: $CHIPMUNK_CONFIG"
    exit 1
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] limit 必须为非负整数"
    exit 1
fi

# 根据 backend 处理环境
PYTHON_EXEC="python"
if [[ "$BACKEND" == "flux" ]]; then
    if [[ -n "$PYTHON_PATH" ]]; then
        PYTHON_EXEC="$PYTHON_PATH"
    else
        if [[ ! -d ".venv" ]]; then
            echo "[ERROR] .venv 虚拟环境不存在，请先创建环境"
            exit 1
        fi

        # shellcheck disable=SC1091
        source .venv/bin/activate || { echo "[ERROR] 激活 .venv 失败"; exit 1; }
        echo "[INFO] 已激活虚拟环境: .venv"

        if [[ -z "${VIRTUAL_ENV:-}" ]]; then
            echo "[ERROR] 虚拟环境未激活，请检查 .venv"
            exit 1
        fi

        EXPECTED_VENV="$PROJECT_ROOT/.venv"
        if [[ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]]; then
            echo "[ERROR] 虚拟环境路径不匹配"
            echo "       当前: $VIRTUAL_ENV"
            echo "       预期: $EXPECTED_VENV"
            echo "       建议:"
            echo "         cd $PROJECT_ROOT && python3.10 -m venv .venv"
            echo "         source .venv/bin/activate && pip install -e \".[all]\""
            exit 1
        fi

        echo "[INFO] 使用虚拟环境: $VIRTUAL_ENV"
    fi
elif [[ "$BACKEND" == "qwen-image" ]]; then
    if [[ -n "$PYTHON_PATH" ]]; then
        PYTHON_EXEC="$PYTHON_PATH"
    else
        TARGET_QWEN_ENV="${QWEN_ENV_OVERRIDE:-$QWEN_ENV_PATH}"
        if [[ -z "$QWEN_ACTIVATE_SCRIPT" || -z "$TARGET_QWEN_ENV" ]]; then
            echo "[ERROR] Qwen-Image 环境未配置。"
            echo "       推荐方式：直接用 --python 传入 Qwen 环境的解释器路径。"
            echo "       备选方式：设置 QWEN_IMAGE_ACTIVATE_SCRIPT 与 QWEN_IMAGE_ENV_PATH（或用 --qwen_env 覆盖）。"
            exit 1
        fi
        if [[ ! -f "$QWEN_ACTIVATE_SCRIPT" ]]; then
            echo "[ERROR] 找不到 Qwen-Image 激活脚本: $QWEN_ACTIVATE_SCRIPT"
            echo "       请通过 --python 或 --qwen_env 指定正确环境"
            exit 1
        fi
        if [[ ! -d "$TARGET_QWEN_ENV" ]]; then
            echo "[ERROR] 找不到 Qwen-Image 环境目录: $TARGET_QWEN_ENV"
            echo "       请通过 --qwen_env 指定，或使用 --python 传入解释器"
            exit 1
        fi
        echo "[INFO] 激活 Qwen-Image 环境: $TARGET_QWEN_ENV"
        # shellcheck disable=SC1090
        source "$QWEN_ACTIVATE_SCRIPT" "$TARGET_QWEN_ENV" || {
            echo "[ERROR] 激活 Qwen-Image 环境失败"
            exit 1
        }
        PYTHON_EXEC="python"
    fi
else
    if [[ -n "$PYTHON_PATH" ]]; then
        PYTHON_EXEC="$PYTHON_PATH"
    else
        PYTHON_EXEC="python"
    fi
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

count_existing_images() {
    local dir="$1"
    local max_idx=-1
    local file
    shopt -s nullglob
    for file in "$dir"/img_*.jpg "$dir"/img_*.png; do
        local base
        base="$(basename "$file")"
        local num="${base#img_}"
        num="${num%%.*}"
        if [[ "$num" =~ ^[0-9]+$ ]]; then
            if (( num > max_idx )); then
                max_idx="$num"
            fi
        fi
    done
    shopt -u nullglob
    if (( max_idx >= 0 )); then
        echo $((max_idx + 1))
    else
        echo 0
    fi
}

# 设置模型目录（仅 flux 后端需要）
if [[ "$BACKEND" == "flux" ]]; then
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
else
    AUTO_MODEL_DIR=""
fi

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="auto_$(date +%Y%m%d_%H%M%S)"
    AUTO_RUN_NAME=true
fi

MODE_LOWER="${MODE,,}"
if [[ "$BACKEND" == "flux" || "$BACKEND" == "chipmunk" ]]; then
    STAGE_LABEL="$MODE_LOWER"
else
    BACKEND_TAG="${BACKEND//-/_}"
    STAGE_LABEL="${BACKEND_TAG}_${MODE_LOWER}"
fi

if [[ -n "$RUN_NAME" ]]; then
    MERGED_ROOT="$BASE_OUTPUT_DIR/${STAGE_LABEL}_$RUN_NAME"
else
    MERGED_ROOT="$BASE_OUTPUT_DIR/${STAGE_LABEL}"
fi

# 统一的聚合输出目录：仅保留 mode 作为一级目录，参数合并为子目录名
if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    NUM_STEPS=4
fi
if [[ "$BACKEND" == "flux" ]]; then
    PARAM_TAG="mn_${MODEL_NAME}_i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_hs_${HICACHE_SCALE_FACTOR}"
elif [[ "$BACKEND" == "chipmunk" ]]; then
    if [[ "$MODE_LOWER" == "chipmunk" || "$MODE_LOWER" == "original" ]]; then
        PARAM_TAG="mn_${MODEL_NAME}_w_${WIDTH}_h_${HEIGHT}_s_${NUM_STEPS}_g_${GUIDANCE}"
    else
        PARAM_TAG="mn_${MODEL_NAME}_w_${WIDTH}_h_${HEIGHT}_s_${NUM_STEPS}_g_${GUIDANCE}_mode_${MODE_LOWER}_i_${INTERVAL}_o_${MAX_ORDER}_hs_${HICACHE_SCALE_FACTOR}"
    fi
else
    PARAM_TAG="i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_hs_${HICACHE_SCALE_FACTOR}"
fi
MERGED_OUTPUT_DIR="$MERGED_ROOT/${PARAM_TAG}"
START_OFFSET=0

if [[ "$FORCE" == true && -d "$MERGED_OUTPUT_DIR" ]]; then
    echo "[FORCE] 移除已存在的输出目录: $MERGED_OUTPUT_DIR"
    rm -rf "$MERGED_OUTPUT_DIR"
elif [[ "$FORCE" != true && -d "$MERGED_OUTPUT_DIR" ]]; then
    START_OFFSET=$(count_existing_images "$MERGED_OUTPUT_DIR")
    if (( START_OFFSET > 0 )); then
        echo "[RESUME] 检测到已有 $START_OFFSET 张图像，将从该位置继续生成"
    fi
fi

TEMP_PROMPT_FILE=$(mktemp "$PROJECT_ROOT/RUN/tmp_multi_gpu_launcher_prompts.XXXXXX.txt") || {
    echo "[ERROR] 创建临时 Prompt 文件失败"
    exit 1
}
REPORT_PATH=$(mktemp "$PROJECT_ROOT/RUN/tmp_multi_gpu_launcher_report.XXXXXX.json") || {
    echo "[ERROR] 创建临时报告文件失败"
    rm -f "$TEMP_PROMPT_FILE"
    exit 1
}

cleanup_tmp_files() {
    rm -f "$TEMP_PROMPT_FILE" "$REPORT_PATH"
}
trap cleanup_tmp_files EXIT

REMAINING_LIMIT=$((LIMIT - START_OFFSET))
if (( REMAINING_LIMIT <= 0 )); then
    echo "[INFO] 目标目录已包含 $START_OFFSET 张图像，达到 limit=$LIMIT，跳过执行。"
    exit 0
fi

START_LINE=$((START_OFFSET + 1))
tail -n +"$START_LINE" "$PROMPT_FILE" | head -n "$REMAINING_LIMIT" > "$TEMP_PROMPT_FILE"

if [[ ! -s "$TEMP_PROMPT_FILE" ]]; then
    echo "[INFO] 无剩余 Prompt 可供生成，退出。"
    exit 0
fi

# 若用户未显式指定步数，schnell 默认步数为 4
if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    NUM_STEPS=4
fi

# 汇总传递给后端的参数
if [[ "$BACKEND" == "flux" ]]; then
    SAMPLE_ARGS=(
        --interval "$INTERVAL"
        --max_order "$MAX_ORDER"
        --first_enhance "$FIRST_ENHANCE"
        --width "$WIDTH"
        --height "$HEIGHT"
        --num_steps "$NUM_STEPS"
        --limit "$LIMIT"
        --hicache_scale "$HICACHE_SCALE_FACTOR"
        --model_name "$MODEL_NAME"
    )

    if [[ "$MODE" == "ClusCa" || "$MODE" == "Hi-ClusCa" ]]; then
        SAMPLE_ARGS+=(
            --clusca_fresh_threshold "$CLUSCA_FRESH_THRESHOLD"
            --clusca_cluster_num "$CLUSCA_CLUSTER_NUM"
            --clusca_cluster_method "$CLUSCA_CLUSTER_METHOD"
            --clusca_k "$CLUSCA_K"
            --clusca_propagation_ratio "$CLUSCA_PROPAGATION_RATIO"
        )
    fi
elif [[ "$BACKEND" == "chipmunk" ]]; then
    SAMPLE_ARGS=(
        --name "$MODEL_NAME"
        --width "$WIDTH"
        --height "$HEIGHT"
        --num_steps "$NUM_STEPS"
        --guidance "$GUIDANCE"
        --chipmunk_config "$CHIPMUNK_CONFIG"
        --quiet True
    )
    if [[ "$MODE_LOWER" != "chipmunk" && "$MODE_LOWER" != "original" ]]; then
        SAMPLE_ARGS+=(
            --cache_mode "$MODE"
            --interval "$INTERVAL"
            --max_order "$MAX_ORDER"
            --first_enhance "$FIRST_ENHANCE"
            --hicache_scale "$HICACHE_SCALE_FACTOR"
        )
    fi
else
    SAMPLE_ARGS=(
        --interval "$INTERVAL"
        --max_order "$MAX_ORDER"
        --first_enhance "$FIRST_ENHANCE"
        --width "$WIDTH"
        --height "$HEIGHT"
        --num_steps "$NUM_STEPS"
        --hicache_scale "$HICACHE_SCALE_FACTOR"
    )
    if [[ -n "$QWEN_MODEL_PATH" ]]; then
        SAMPLE_ARGS+=(--model_path "$QWEN_MODEL_PATH")
    fi
fi

if [[ ${#EXTRA_SAMPLE_ARGS[@]} -gt 0 ]]; then
    SAMPLE_ARGS+=("${EXTRA_SAMPLE_ARGS[@]}")
fi

if [[ "$BACKEND" == "flux" && -n "$AUTO_MODEL_DIR" ]]; then
    SAMPLE_ARGS+=(--model_dir "$AUTO_MODEL_DIR")
fi

# 展示配置概要
echo "================================="
echo "多卡采样配置:"
echo "后端: $BACKEND"
echo "模式: $MODE"
if [[ "$BACKEND" == "flux" ]]; then
    echo "FLUX 模型: $MODEL_NAME"
elif [[ "$BACKEND" == "qwen-image" ]]; then
    if [[ -n "$QWEN_MODEL_PATH" ]]; then
        echo "Qwen 模型路径: $QWEN_MODEL_PATH"
    else
        echo "Qwen 模型路径: (默认)"
    fi
else
    echo "Chipmunk 模型: $MODEL_NAME"
fi
if [[ -n "$GPU_LIST" ]]; then
    echo "GPU 列表: $GPU_LIST"
elif [[ -n "$NUM_GPUS" ]]; then
    echo "GPU 数量: $NUM_GPUS (从 0 开始)"
else
    echo "GPU: 自动检测"
fi
echo "基础输出目录: $BASE_OUTPUT_DIR"
if [[ "$AUTO_RUN_NAME" == true && -z "$RUN_NAME" ]]; then
    echo "运行名: (默认，相同参数复用路径)"
elif [[ "$AUTO_RUN_NAME" == true ]]; then
    echo "运行名: $RUN_NAME (自动生成)"
else
    echo "运行名: $RUN_NAME"
fi
echo "Prompt 文件: $PROMPT_FILE (限制 $LIMIT 条)"
echo "临时 Prompt: $TEMP_PROMPT_FILE"
if (( START_OFFSET > 0 )); then
    echo "恢复模式: 起始索引偏移 $START_OFFSET，预计再生成 $REMAINING_LIMIT 张"
else
    echo "计划生成: $REMAINING_LIMIT 张 (limit=$LIMIT)"
fi
echo "尺寸: ${WIDTH}x${HEIGHT}, 步数: $NUM_STEPS"
if [[ "$BACKEND" == "flux" ]]; then
    if [[ -n "$MODEL_DIR" ]]; then
        echo "模型目录: $MODEL_DIR"
    elif [[ -n "$AUTO_MODEL_DIR" ]]; then
        echo "自动检测模型目录: $AUTO_MODEL_DIR"
    fi
fi
if [[ "$BACKEND" == "flux" || "$BACKEND" == "qwen-image" ]]; then
    echo "间隔: $INTERVAL, 阶数: $MAX_ORDER"
    echo "HiCache 缩放: $HICACHE_SCALE_FACTOR"
elif [[ "$BACKEND" == "chipmunk" && "$MODE_LOWER" != "chipmunk" && "$MODE_LOWER" != "original" ]]; then
    echo "Chipmunk 缓存参数: interval=$INTERVAL, max_order=$MAX_ORDER, first_enhance=$FIRST_ENHANCE"
    echo "HiCache 缩放: $HICACHE_SCALE_FACTOR"
fi
if [[ "$BACKEND" == "chipmunk" ]]; then
    echo "Chipmunk guidance: $GUIDANCE"
    echo "Chipmunk 配置: $CHIPMUNK_CONFIG"
fi
if [[ "$BACKEND" == "flux" && "$MODE" == "ClusCa" ]]; then
    echo "ClusCa fresh 阈值: $CLUSCA_FRESH_THRESHOLD"
    echo "ClusCa 聚类: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "ClusCa k: $CLUSCA_K, 传播比例: $CLUSCA_PROPAGATION_RATIO"
elif [[ "$BACKEND" == "flux" && "$MODE" == "Hi-ClusCa" ]]; then
    echo "Hi-ClusCa fresh 阈值: $CLUSCA_FRESH_THRESHOLD"
    echo "Hi-ClusCa 聚类: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "Hi-ClusCa k: $CLUSCA_K, 传播比例: $CLUSCA_PROPAGATION_RATIO"
    echo "Hi-ClusCa HiCache 缩放: $HICACHE_SCALE_FACTOR"
fi
echo "保留临时分片: $KEEP_TEMP"
echo "干运行: $DRY_RUN"
if [[ ${#EXTRA_SAMPLE_ARGS[@]} -gt 0 ]]; then
    printf '额外 sample.sh 参数: %s\n' "${EXTRA_SAMPLE_ARGS[*]}"
fi
echo "================================="

PYTHON_CMD=(
    "$PYTHON_EXEC" "RUN/multi_gpu_launcher.py"
    --backend "$BACKEND"
    --mode "$MODE"
    --prompt-file "$TEMP_PROMPT_FILE"
    --full-prompt-file "$PROMPT_FILE"
    --base-output-dir "$BASE_OUTPUT_DIR"
    --run-name "$RUN_NAME"
    --report-path "$REPORT_PATH"
    --start-offset "$START_OFFSET"
)

if [[ "$BACKEND" == "chipmunk" ]]; then
    PYTHON_CMD+=(--chipmunk-param-tag "$PARAM_TAG")
fi

if [[ -n "$GPU_LIST" ]]; then
    PYTHON_CMD+=(--gpus "$GPU_LIST")
fi
if [[ -n "$NUM_GPUS" ]]; then
    PYTHON_CMD+=(--num-gpus "$NUM_GPUS")
fi
if [[ "$KEEP_TEMP" == true ]]; then
    PYTHON_CMD+=(--keep-temp)
fi
if [[ "$DRY_RUN" == true ]]; then
    PYTHON_CMD+=(--dry-run)
fi

PYTHON_CMD+=(--)
if [[ ${#SAMPLE_ARGS[@]} -gt 0 ]]; then
    PYTHON_CMD+=("${SAMPLE_ARGS[@]}")
fi

echo "[INFO] 启动多卡采样..."
printf '[CMD] %s\n' "${PYTHON_CMD[*]}"

"${PYTHON_CMD[@]}"
PYTHON_EXIT_CODE=$?

if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] 多卡采样执行失败 (退出码: $PYTHON_EXIT_CODE)"
    exit $PYTHON_EXIT_CODE
fi

echo "多卡采样完成！"
FINAL_OUTPUT_DIR="$MERGED_OUTPUT_DIR"
if [[ -f "$REPORT_PATH" ]]; then
    unset report_success report_path
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            eval "$line"
        fi
    done < <(
        python - <<'PY' "$REPORT_PATH"
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

success = bool(data.get("success"))
path = data.get("final_output_path") or ""

print(f"report_success={str(success).lower()}")
print("report_path=" + shlex.quote(path))
PY
    )

    if [[ "${report_success:-false}" == "true" && -n "${report_path:-}" ]]; then
        FINAL_OUTPUT_DIR="$report_path"
    elif [[ -n "${report_path:-}" ]]; then
        FINAL_OUTPUT_DIR="$report_path"
        echo "[WARN] 报告文件标记 success=false，仍使用解析出的目录: $FINAL_OUTPUT_DIR"
    else
        echo "[WARN] 未能从报告文件解析最终输出目录，使用默认值: $FINAL_OUTPUT_DIR"
    fi
else
    echo "[WARN] 未找到报告文件: $REPORT_PATH"
fi

echo "结果聚合目录: $MERGED_ROOT"
echo "最终图像目录: $FINAL_OUTPUT_DIR"

if [[ "$BACKEND" == "flux" ]]; then
    echo "================================="
    # 固定 GT 建议目录为 Taylor baseline interval_1/order_2
    GT_SUGGEST="$PROJECT_ROOT/results/taylor/interval_1/order_2"
    echo "后续可执行评估命令示例:"
    echo "  bash $PROJECT_ROOT/evaluation/run_eval.sh --acc \"multi=$FINAL_OUTPUT_DIR\" --gt \"$GT_SUGGEST\""
    echo "================================="
fi

trap - EXIT
cleanup_tmp_files
