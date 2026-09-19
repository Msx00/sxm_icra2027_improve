#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/weights"
hf download runwayml/stable-diffusion-inpainting --local-dir "$ROOT/weights/sd15_inpainting"
