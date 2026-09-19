#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DISTRISURG_PYTHON:-/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11}"
GPU_IDS="${GPU_IDS:-0,1,3}"
EVAL_GPU_ID="${EVAL_GPU_ID:-${GPU_IDS%%,*}}"
CONFIG="${CONFIG:-$ROOT/configs/distrisurg_dataset89.yaml}"
SUITE="${SUITE:-core}"
PHASE="${PHASE:-train}"
SEEDS="${SEEDS:-6666}"
STEPS="${STEPS:-}"
SAVE_EVERY="${SAVE_EVERY:-}"
MAX_FRAMES="${MAX_FRAMES:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

declare -A ABLATION_CONFIGS=(
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

core=(
  full no_depth_completion deterministic_depth isotropic_footprint
  no_physics_router no_hard_composition no_rgbd_cycle no_uncertainty_loss
  no_trust_cleanup
)
renderer=(hard_point soft_splat dss_no_router)
dss=(dss_1_sample full dss_5_samples isotropic_footprint)
all=(
  full no_depth_completion deterministic_depth isotropic_footprint
  no_physics_router no_hard_composition no_rgbd_cycle no_uncertainty_loss
  no_trust_cleanup hard_point soft_splat dss_no_router dss_5_samples
)

usage() {
  cat <<'EOF'
Run reproducible DistriSurg ablations sequentially.

Environment variables:
  SUITE=core|renderer|dss|all   Experiment group (default: core)
  EXPERIMENTS=a,b,c            Explicit names; overrides SUITE
  PHASE=train|eval|all         Train, zero-shot evaluate, or both
  SEEDS=6666,7777,8888         Comma-separated training seeds
  GPU_IDS=0,1,3                GPUs used by distributed training
  EVAL_GPU_ID=0                Single GPU used by evaluation
  STEPS=10000                  Optional override of optimizer steps
  SAVE_EVERY=500               Optional checkpoint interval override
  MAX_FRAMES=10                Optional evaluation smoke-test cap
  SKIP_COMPLETED=1             Skip manifests marked completed
  DRY_RUN=1                    Print commands without executing

Examples:
  bash run_ablation.sh --list
  SUITE=core GPU_IDS=0,1,3 bash run_ablation.sh
  EXPERIMENTS=full,no_rgbd_cycle PHASE=all bash run_ablation.sh
  SUITE=core SEEDS=6666,7777,8888 PHASE=train bash run_ablation.sh
EOF
}

list_experiments() {
  printf '%s\n' "${!ABLATION_CONFIGS[@]}" | sort
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--list" ]]; then
  list_experiments
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

case "$PHASE" in
  train|eval|all) ;;
  *) echo "PHASE must be train, eval, or all; got: $PHASE" >&2; exit 2 ;;
esac

if [[ -n "${EXPERIMENTS:-}" ]]; then
  IFS=',' read -r -a experiments <<< "$EXPERIMENTS"
else
  case "$SUITE" in
    core) experiments=("${core[@]}") ;;
    renderer) experiments=("${renderer[@]}") ;;
    dss) experiments=("${dss[@]}") ;;
    all) experiments=("${all[@]}") ;;
    *) echo "Unknown SUITE: $SUITE" >&2; exit 2 ;;
  esac
fi
IFS=',' read -r -a seeds <<< "$SEEDS"

for name in "${experiments[@]}"; do
  [[ -v "ABLATION_CONFIGS[$name]" ]] || {
    echo "Unknown experiment '$name'. Available names:" >&2
    list_experiments >&2
    exit 2
  }
  cfg="${ABLATION_CONFIGS[$name]}"
  [[ -z "$cfg" || -f "$cfg" ]] || { echo "Missing config: $cfg" >&2; exit 2; }
done

is_complete() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("completed") else 1)' "$manifest"
}

run_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

train_one() {
  local name="$1" seed="$2" cfg="$3" output="$4"
  if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$output/training_manifest.json"; then
    echo "[skip train] $name seed=$seed is complete"
    return
  fi
  local resume=""
  [[ ! -f "$output/latest.pt" ]] || resume="$output/latest.pt"
  local args=(--set "train.seed=$seed")
  [[ -z "$STEPS" ]] || args+=(--set "train.steps=$STEPS")
  [[ -z "$SAVE_EVERY" ]] || args+=(--set "train.save_every=$SAVE_EVERY")
  echo "[train] $name seed=$seed config=${cfg:-full} output=$output"
  if [[ "$DRY_RUN" == "1" ]]; then
    run_or_print env GPU_IDS="$GPU_IDS" CONFIG="$CONFIG" TRAIN_OUTPUT="$output" \
      ABLATION_CONFIG="$cfg" RESUME="$resume" bash "$ROOT/run_train.sh" "${args[@]}"
  else
    mkdir -p "$output"
    env GPU_IDS="$GPU_IDS" CONFIG="$CONFIG" TRAIN_OUTPUT="$output" \
      ABLATION_CONFIG="$cfg" RESUME="$resume" bash "$ROOT/run_train.sh" "${args[@]}" \
      2>&1 | tee -a "$output/console.log"
  fi
}

eval_one() {
  local name="$1" seed="$2" cfg="$3" checkpoint="$4" output="$5"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for $name seed=$seed: $checkpoint" >&2
    return 1
  fi
  if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$output/run_manifest.json"; then
    echo "[skip eval] $name seed=$seed is complete"
    return
  fi
  local args=()
  [[ -z "$MAX_FRAMES" ]] || args+=(--max-frames "$MAX_FRAMES")
  echo "[eval] $name seed=$seed config=${cfg:-full} output=$output"
  run_or_print env GPU_ID="$EVAL_GPU_ID" CONFIG="$CONFIG" CHECKPOINT="$checkpoint" \
    OUTPUT="$output" ABLATION_CONFIG="$cfg" OVERWRITE=1 \
    bash "$ROOT/run_dataset89.sh" "${args[@]}"
}

train_outputs=()
eval_outputs=()
labels=()
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
  for name in "${experiments[@]}"; do
    cfg="${ABLATION_CONFIGS[$name]}"
    train_output="$ROOT/checkpoints/ablations/$name/seed_$seed"
    eval_output="$ROOT/outputs/ablations/$name/seed_$seed"
    if [[ "$PHASE" == "train" || "$PHASE" == "all" ]]; then
      train_one "$name" "$seed" "$cfg" "$train_output"
    fi
    if [[ "$PHASE" == "eval" || "$PHASE" == "all" ]]; then
      eval_one "$name" "$seed" "$cfg" "$train_output/latest.pt" "$eval_output"
      eval_outputs+=("$eval_output")
      labels+=("$name-s$seed")
    fi
    train_outputs+=("$train_output")
  done
done

if [[ "$PHASE" != "train" && "$DRY_RUN" != "1" && ${#eval_outputs[@]} -gt 0 ]]; then
  table="$ROOT/outputs/ablations/comparison_${SUITE}"
  "$PYTHON" "$ROOT/scripts/compare_runs.py" "${eval_outputs[@]}" \
    --labels "${labels[@]}" --output "$table"
  echo "[done] comparison: ${table}.md"
else
  echo "[done] phase=$PHASE experiments=${#experiments[@]} seeds=${#seeds[@]}"
fi
