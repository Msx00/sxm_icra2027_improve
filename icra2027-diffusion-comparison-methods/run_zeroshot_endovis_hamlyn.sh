#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${COMPARISON_PYTHON:-/usr/bin/python3}"
DATA_ROOT="${ZERO_DATA_ROOT:-/home/data/mashixing/dataset_18tb/icra2027-diffusion-zero-test-dataset/endovis/dataset89}"
SCENES_FILE="${ZERO_SCENES_FILE:-$ROOT/zeroshot_endovis_dataset89_scenes.txt}"
OUTPUT_ROOT="${ZERO_OUTPUT_ROOT:-/home/data/mashixing/dataset_8tb/iMed/comparison/task2-icra/cross_endo_rendering_surpvised/results/zeroshot/endovis_dataset89_overlap_gt_0p5_neighbor_fill64_comparison}"
GPU_ID="${GPU_ID:-3}"
MIN_VALID_RATIO="${MIN_VALID_RATIO:-0.5}"
SMALL_HOLE_MAX_AREA="${SMALL_HOLE_MAX_AREA:-0}"
SEAM_KERNEL="${SEAM_KERNEL:-1}"

extra_args=()
if [[ -n "${METHODS:-}" ]]; then
  read -r -a selected_methods <<< "$METHODS"
  extra_args+=(--methods "${selected_methods[@]}")
fi
[[ -z "${MAX_FRAMES:-}" ]] || extra_args+=(--max-frames "$MAX_FRAMES")
[[ "${OVERWRITE_WARPS:-0}" != "1" ]] || extra_args+=(--overwrite-warps)
[[ "${OVERWRITE_INFERENCE:-0}" != "1" ]] || extra_args+=(--overwrite-inference)
[[ "${SKIP_EVALUATION:-0}" != "1" ]] || extra_args+=(--skip-evaluation)
[[ "${LPIPS:-0}" != "1" ]] || extra_args+=(--lpips)

exec "$PYTHON" "$ROOT/run_zeroshot_endovis_hamlyn.py" \
  --data-root "$DATA_ROOT" --scenes-file "$SCENES_FILE" \
  --output-root "$OUTPUT_ROOT" --gpu-id "$GPU_ID" \
  --min-valid-ratio "$MIN_VALID_RATIO" \
  --small-hole-max-area "$SMALL_HOLE_MAX_AREA" \
  --seam-kernel "$SEAM_KERNEL" \
  "${extra_args[@]}" "$@"
