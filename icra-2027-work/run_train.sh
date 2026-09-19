#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DISTRISURG_PYTHON:-/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11}"
GPU_ID="${GPU_ID:-0}"
OUTPUT="${TRAIN_OUTPUT:-$ROOT/checkpoints/distrisurg_main}"
CONFIG="${CONFIG:-$ROOT/configs/distrisurg_dataset89.yaml}"

extra=()
[[ -z "${RESUME:-}" ]] || extra+=(--resume "$RESUME")
[[ -z "${ABLATION_CONFIG:-}" ]] || extra+=(--ablation-config "$ABLATION_CONFIG")

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/train.py \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --device cuda \
  "${extra[@]}" \
  "$@"

