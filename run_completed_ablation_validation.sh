#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DISTRISURG_PYTHON:-/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
SEED="${SEED:-6666}"
VAL_ROOT="${VAL_ROOT:-/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs}"
VAL_SCENES="${VAL_SCENES:-/home/data/mashixing/dataset_8tb/iMed/comparison/task2-icra/val_scenes.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/ablations_validation_no_post}"
SMALL_HOLE_MAX_AREA_RATIO="${SMALL_HOLE_MAX_AREA_RATIO:-0}"
SMALL_HOLE_INPAINT_RADIUS="${SMALL_HOLE_INPAINT_RADIUS:-3.0}"

declare -A CONFIGS=(
  [full]=""
  [no_depth_completion]="$ROOT/configs/ablations/a1_no_depth_completion.yaml"
  [deterministic_depth]="$ROOT/configs/ablations/a2_deterministic_depth.yaml"
  [hard_point]="$ROOT/configs/ablations/a3_hard_point.yaml"
  [soft_splat]="$ROOT/configs/ablations/a4_soft_splat.yaml"
  [dss_no_router]="$ROOT/configs/ablations/a5_dss_no_router.yaml"
  [no_physics_router]="$ROOT/configs/ablations/a5b_no_physics_router.yaml"
  [no_hard_composition]="$ROOT/configs/ablations/a6_no_hard_composition.yaml"
  [no_rgbd_cycle]="$ROOT/configs/ablations/a7_no_rgbd_cycle.yaml"
  [isotropic_footprint]="$ROOT/configs/ablations/a8_isotropic_footprint.yaml"
  [no_uncertainty_loss]="$ROOT/configs/ablations/a9_no_uncertainty_loss.yaml"
  [dss_1_sample]="$ROOT/configs/ablations/a10_dss_1_sample.yaml"
  [dss_5_samples]="$ROOT/configs/ablations/a11_dss_5_samples.yaml"
  [no_trust_cleanup]="$ROOT/configs/ablations/a12_no_trust_cleanup.yaml"
)
ORDER=(
  full no_depth_completion deterministic_depth isotropic_footprint
  no_physics_router no_hard_composition no_rgbd_cycle no_uncertainty_loss
  no_trust_cleanup hard_point soft_splat dss_no_router dss_1_sample dss_5_samples
)

is_complete_manifest() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  "$PYTHON" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("completed") else 1)' \
    "$manifest"
}

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  [[ "$WAIT_FOR_PID" =~ ^[0-9]+$ ]] || {
    echo "WAIT_FOR_PID must be numeric" >&2
    exit 2
  }
  echo "Waiting for training coordinator PID $WAIT_FOR_PID before validation"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
  done
fi

[[ -d "$VAL_ROOT" ]] || { echo "Missing validation root: $VAL_ROOT" >&2; exit 1; }
[[ -f "$VAL_SCENES" ]] || { echo "Missing validation scenes: $VAL_SCENES" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

runs=()
labels=()
job_names=()
job_checkpoints=()
job_outputs=()
for name in "${ORDER[@]}"; do
  checkpoint_dir="$ROOT/checkpoints/ablations/$name/seed_$SEED"
  if ! is_complete_manifest "$checkpoint_dir/training_manifest.json"; then
    echo "[skip] $name seed=$SEED is not completed"
    continue
  fi
  checkpoint="$checkpoint_dir/latest.pt"
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  output="$OUTPUT_ROOT/$name/seed_$SEED"
  job_names+=("$name")
  job_checkpoints+=("$checkpoint")
  job_outputs+=("$output")
  runs+=("$output")
  labels+=("$name-s$SEED")
done

(( ${#runs[@]} > 0 )) || { echo "No completed ablations found" >&2; exit 1; }

IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
for gpu_id in "${gpu_ids[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu_id" >&2; exit 2; }
done

worker() {
  local worker_index="$1" gpu_id="$2" job_index name checkpoint output
  for ((job_index = worker_index; job_index < ${#job_names[@]}; job_index += ${#gpu_ids[@]})); do
    name="${job_names[$job_index]}"
    checkpoint="${job_checkpoints[$job_index]}"
    output="${job_outputs[$job_index]}"
    if is_complete_manifest "$output/run_manifest.json"; then
      echo "[reuse] completed validation: $name"
      continue
    fi
    echo "[validate] $name seed=$SEED GPU=$gpu_id"
    extra=()
    [[ -z "${CONFIGS[$name]}" ]] || extra+=(--ablation-config "${CONFIGS[$name]}")
    CHECKPOINT="$checkpoint" \
    VAL_ROOT="$VAL_ROOT" \
    VAL_SCENES="$VAL_SCENES" \
    OUTPUT="$output" \
    GPU_ID="$gpu_id" \
    OVERWRITE=1 \
    SMALL_HOLE_MAX_AREA_RATIO="$SMALL_HOLE_MAX_AREA_RATIO" \
    SMALL_HOLE_INPAINT_RADIUS="$SMALL_HOLE_INPAINT_RADIUS" \
      bash "$ROOT/run_validation.sh" "${extra[@]}"
  done
}

worker_pids=()
for worker_index in "${!gpu_ids[@]}"; do
  worker "$worker_index" "${gpu_ids[$worker_index]}" &
  worker_pids+=("$!")
done
failed=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "$worker_pid"; then
    failed=1
  fi
done
[[ "$failed" == "0" ]] || { echo "One or more validation workers failed" >&2; exit 1; }

table="$OUTPUT_ROOT/comparison_completed_seed_$SEED"
"$PYTHON" "$ROOT/scripts/compare_runs.py" "${runs[@]}" \
  --labels "${labels[@]}" --output "$table"
echo "[done] ${table}.md"
