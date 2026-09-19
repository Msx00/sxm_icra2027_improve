#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/weights"
echo "MAT official checkpoint hosting requires browser access."
echo "Download Places_512_FullData.pkl from the official README link into:"
echo "$ROOT/weights/Places_512_FullData.pkl"
