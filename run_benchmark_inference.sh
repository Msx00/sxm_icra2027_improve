#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
COMPARISON_ROOT="${COMPARISON_ROOT:-/home/data/mashixing/dataset_18tb/icra2027-diffusion-comparison-methods}"
SUPERVISED_ROOT="${SUPERVISED_ROOT:-/home/data/mashixing/dataset_8tb/iMed/comparison/task2-icra/cross_endo_rendering_surpvised}"
IMED_ROOT="${IMED_ROOT:-/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs}"
ENDOVIS_ROOT="${ENDOVIS_ROOT:-/home/data/mashixing/dataset_18tb/icra2027-diffusion-zero-test-dataset/endovis/all}"
VAL_SCENES="${VAL_SCENES:-$ROOT/configs/val_scenes.txt}"
ZERO_SCENES="${ZERO_SCENES:-$ROOT/configs/endovis_all_scenes.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/distrisurg_main/latest.pt}"
DOMAIN_LORA="${DOMAIN_LORA:-$SUPERVISED_ROOT/myTrain/endo1-supervised-rank32/checkpoint-002500}"
LORA_RESULT_NAME="${LORA_RESULT_NAME:-endo1_supervised_rank32_step2500}"
GPU_ID="${GPU_ID:-0}"
SPLITS="${SPLITS:-validation,zeroshot}"
METHODS="${METHODS:-distrisurg,sd15,lama,supervised_lora}"
MAX_FRAMES="${MAX_FRAMES:-}"
OVERWRITE="${OVERWRITE:-0}"

contains() { [[ ",$1," == *",$2,"* ]]; }

for path in "$COMPARISON_ROOT" "$SUPERVISED_ROOT" "$IMED_ROOT" "$ENDOVIS_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 2; }
done
for path in "$VAL_SCENES" "$ZERO_SCENES" "$CHECKPOINT" \
  "$DOMAIN_LORA/pytorch_lora_weights.safetensors"; do
  [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 2; }
done

generic_methods=()
contains "$METHODS" sd15 && generic_methods+=(sd15)
contains "$METHODS" lama && generic_methods+=(lama)
contains "$METHODS" mat && generic_methods+=(mat)

common_optional=()
[[ -z "$MAX_FRAMES" ]] || common_optional+=(MAX_FRAMES="$MAX_FRAMES")

run_validation() {
  local destination="$RESULTS_ROOT/validation"
  mkdir -p "$destination"
  if contains "$METHODS" distrisurg; then
    echo "[validation] DistriSurg"
    env GPU_ID="$GPU_ID" CHECKPOINT="$CHECKPOINT" \
      OUTPUT="$destination/distrisurg" MIN_OVERLAP=0.0 OVERWRITE="$OVERWRITE" \
      "${common_optional[@]}" bash "$ROOT/run_validation.sh"
  fi
  if (( ${#generic_methods[@]} > 0 )); then
    echo "[validation] Generic inpainting: ${generic_methods[*]}"
    env GPU_ID="$GPU_ID" ZERO_DATA_ROOT="$IMED_ROOT" \
      ZERO_SCENES_FILE="$VAL_SCENES" ZERO_OUTPUT_ROOT="$destination/generic_inpainting" \
      MIN_VALID_RATIO=0.0 SMALL_HOLE_MAX_AREA=0 SEAM_KERNEL=1 \
      OVERWRITE_WARPS="$OVERWRITE" OVERWRITE_INFERENCE="$OVERWRITE" \
      METHODS="${generic_methods[*]}" "${common_optional[@]}" \
      bash "$COMPARISON_ROOT/run_zeroshot_endovis_hamlyn.sh"
  fi
  if contains "$METHODS" supervised_lora; then
    echo "[validation] Supervised LoRA"
    env DOMAIN_LORA="$DOMAIN_LORA" RESULT_NAME="$LORA_RESULT_NAME" \
      VAL_SCENES_FILE="$VAL_SCENES" SMALL_HOLE_MAX_AREA=0 SEAM_KERNEL=1 \
      MIN_VALID_RATIO=0.0 OVERWRITE="$OVERWRITE" PREPARE_OVERWRITE="$OVERWRITE" \
      "${common_optional[@]}" \
      bash "$SUPERVISED_ROOT/run_all_finetune.sh" \
      "$IMED_ROOT" "$destination/supervised_lora" "$GPU_ID"
  fi
}

run_zeroshot() {
  local destination="$RESULTS_ROOT/zeroshot"
  mkdir -p "$destination"
  if contains "$METHODS" distrisurg; then
    echo "[zeroshot] DistriSurg"
    env GPU_ID="$GPU_ID" CHECKPOINT="$CHECKPOINT" \
      OUTPUT="$destination/distrisurg" SMALL_HOLE_MAX_AREA_RATIO=0.02 \
      OVERWRITE="$OVERWRITE" "${common_optional[@]}" \
      bash "$ROOT/run_dataset89.sh" \
      --set "data.eval_root=$ENDOVIS_ROOT" \
      --scenes-file "$ZERO_SCENES" --split-name endovis_all
  fi
  if (( ${#generic_methods[@]} > 0 )); then
    echo "[zeroshot] Generic inpainting: ${generic_methods[*]}"
    env GPU_ID="$GPU_ID" ZERO_DATA_ROOT="$ENDOVIS_ROOT" \
      ZERO_SCENES_FILE="$ZERO_SCENES" ZERO_OUTPUT_ROOT="$destination/generic_inpainting" \
      MIN_VALID_RATIO=0.5 SMALL_HOLE_MAX_AREA=0 SEAM_KERNEL=1 \
      OVERWRITE_WARPS="$OVERWRITE" OVERWRITE_INFERENCE="$OVERWRITE" \
      METHODS="${generic_methods[*]}" "${common_optional[@]}" \
      bash "$COMPARISON_ROOT/run_zeroshot_endovis_hamlyn.sh"
  fi
  if contains "$METHODS" supervised_lora; then
    echo "[zeroshot] Supervised LoRA"
    env GPU_ID="$GPU_ID" ZERO_DATA_ROOT="$ENDOVIS_ROOT" \
      ZERO_SCENES_FILE="$ZERO_SCENES" ZERO_OUTPUT_ROOT="$destination/supervised_lora" \
      DOMAIN_LORA="$DOMAIN_LORA" RESULT_NAME="$LORA_RESULT_NAME" \
      MIN_VALID_RATIO=0.5 SMALL_HOLE_MAX_AREA=0 SEAM_KERNEL=1 \
      OVERWRITE_WARPS="$OVERWRITE" OVERWRITE_INFERENCE="$OVERWRITE" \
      "${common_optional[@]}" \
      bash "$SUPERVISED_ROOT/run_zeroshot_endovis_hamlyn.sh"
  fi
}

contains "$SPLITS" validation && run_validation
contains "$SPLITS" zeroshot && run_zeroshot
echo "Completed requested inference under: $RESULTS_ROOT"
