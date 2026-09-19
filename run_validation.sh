#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DISTRISURG_PYTHON:-/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11}"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-$ROOT/configs/distrisurg_dataset89.yaml}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/distrisurg_main/latest.pt}"
VAL_ROOT="${VAL_ROOT:-/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs}"
VAL_SCENES="${VAL_SCENES:-$ROOT/configs/val_scenes.txt}"
OUTPUT="${OUTPUT:-$ROOT/outputs/validation_all_frames}"
# The iMED validation split has about 0.42 raw depth support in normal frames;
# dataset89's >0.5 selection rule would therefore discard valid validation data.
MIN_OVERLAP="${MIN_OVERLAP:-0.0}"
SMALL_HOLE_MAX_AREA_RATIO="${SMALL_HOLE_MAX_AREA_RATIO:-0.02}"
SMALL_HOLE_INPAINT_RADIUS="${SMALL_HOLE_INPAINT_RADIUS:-3.0}"

[[ -f "$CHECKPOINT" ]] || {
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
}
[[ -f "$VAL_SCENES" ]] || {
  echo "Validation scene list not found: $VAL_SCENES" >&2
  exit 2
}

extra=()
[[ "${OVERWRITE:-0}" != "1" ]] || extra+=(--overwrite)
[[ -z "${MAX_FRAMES:-}" ]] || extra+=(--max-frames "$MAX_FRAMES")

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/infer_dataset89.py \
  --config "$CONFIG" \
  --set "data.eval_root=$VAL_ROOT" \
  --set "data.min_overlap=$MIN_OVERLAP" \
  --checkpoint "$CHECKPOINT" \
  --scenes-file "$VAL_SCENES" \
  --split-name validation \
  --output "$OUTPUT" \
  --device cuda \
  --small-hole-max-area-ratio "$SMALL_HOLE_MAX_AREA_RATIO" \
  --small-hole-inpaint-radius "$SMALL_HOLE_INPAINT_RADIUS" \
  "${extra[@]}" \
  "$@"
