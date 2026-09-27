#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="$PWD/.cache/uv"
  uv venv --python 3.11 --allow-existing .venv-data
  uv pip install --python .venv-data/bin/python -r datasets/coco_single/requirements.txt
else
  python3 -m venv .venv-data
  .venv-data/bin/python -m pip install -r datasets/coco_single/requirements.txt
fi
.venv-data/bin/python datasets/coco_single/prepare.py
