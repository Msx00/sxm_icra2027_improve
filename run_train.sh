#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DISTRISURG_PYTHON:-/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11}"
GPU_IDS="${GPU_IDS:-0,1,3}"
OUTPUT="${TRAIN_OUTPUT:-$ROOT/checkpoints/distrisurg_a6_main}"
CONFIG="${CONFIG:-$ROOT/configs/distrisurg_dataset89.yaml}"
ABLATION_CONFIG="${ABLATION_CONFIG:-$ROOT/configs/ablations/a6_no_hard_composition.yaml}"

extra=()
[[ -z "${RESUME:-}" ]] || extra+=(--resume "$RESUME")
if [[ "$ABLATION_CONFIG" != "none" ]]; then
  extra+=(--ablation-config "$ABLATION_CONFIG")
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${#gpu_list[@]}" \
  scripts/train.py \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --device cuda \
  "${extra[@]}" \
  "$@"
