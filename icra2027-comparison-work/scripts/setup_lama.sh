#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/weights"
curl -fL -C - --retry 10 --retry-all-errors --retry-delay 2 -o "$ROOT/weights/big-lama.zip" https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip -o "$ROOT/weights/big-lama.zip" -d "$ROOT/weights"
