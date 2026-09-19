#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DISTRISURG_PYTHON:-/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m unittest discover -s tests -v

