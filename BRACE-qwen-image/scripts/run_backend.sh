#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND="flux"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help)
      cat <<EOF
Usage:
  $0 --backend flux [-- <args forwarded to models/flux/src/sample.py>...]
  $0 --backend qwen-image [-- <args forwarded to python -m models.qwen_image.sample>...]

Options:
  --backend   flux | qwen-image
  --python    Python executable to run the backend (default: \$PYTHON_BIN or 'python')

Notes:
  - Arguments after '--' are forwarded to the selected backend verbatim.
  - For qwen-image, run this script inside your Qwen environment, or pass --python /path/to/python.

Examples:
  # FLUX
  $0 --backend flux -- --prompt_file resources/prompts/prompt.txt --output_dir outputs/flux --cache_mode HiCache

  # Qwen-Image (run with Qwen environment python)
  $0 --backend qwen-image --python /path/to/qwen/python -- --model_path /path/to/Qwen-Image --output_dir outputs/qwen_image
EOF
      exit 0
      ;;
    --)
      shift
      BACKEND_ARGS+=("$@")
      break
      ;;
    *)
      BACKEND_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" models/run_backend.py --backend "$BACKEND" "${BACKEND_ARGS[@]}"
