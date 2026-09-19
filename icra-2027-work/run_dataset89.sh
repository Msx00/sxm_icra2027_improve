#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DISTRISURG_PYTHON:-/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11}"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-$ROOT/configs/distrisurg_dataset89.yaml}"
MODE="${MODE:-full}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/distrisurg_main/latest.pt}"
OUTPUT="${OUTPUT:-$ROOT/outputs/dataset89_overlap_gt_0p5}"

extra=()
if [[ "$MODE" == "renderer" ]]; then
  extra+=(--renderer-only)
elif [[ "$MODE" == "full" ]]; then
  [[ -f "$CHECKPOINT" ]] || {
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Run run_train.sh first, or set MODE=renderer for the untrained DSS baseline." >&2
    exit 2
  }
  extra+=(--checkpoint "$CHECKPOINT")
else
  echo "MODE must be full or renderer, got: $MODE" >&2
  exit 2
fi
[[ -z "${ABLATION_CONFIG:-}" ]] || extra+=(--ablation-config "$ABLATION_CONFIG")
[[ "${OVERWRITE:-0}" != "1" ]] || extra+=(--overwrite)
[[ -z "${MAX_FRAMES:-}" ]] || extra+=(--max-frames "$MAX_FRAMES")

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/infer_dataset89.py \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --device cuda \
  "${extra[@]}" \
  "$@"

