#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for repo in lama MAT ZITS_inpainting Endo-4DGS; do
  printf '%-18s ' "$repo"; git -C "$ROOT/third_party/$repo" rev-parse --short HEAD 2>/dev/null || echo MISSING
done
for item in sd15_inpainting big-lama Places_512_FullData.pkl; do
  test -e "$ROOT/weights/$item" && echo "weight $item: READY" || echo "weight $item: MISSING"
done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: unavailable in this session"
