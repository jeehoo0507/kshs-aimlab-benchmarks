#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash datasets/imagenet/setup.sh /path/to/official-archives [output-directory]" >&2
  exit 2
fi
OUTPUT="${2:-data/imagenet}"

if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="$PWD/.cache/uv"
  uv venv --python 3.11 --allow-existing .venv-data
  uv pip install --python .venv-data/bin/python -r datasets/imagenet/requirements.txt
else
  python3 -m venv .venv-data
  .venv-data/bin/python -m pip install -r datasets/imagenet/requirements.txt
fi

.venv-data/bin/python datasets/imagenet/prepare.py --source "$1" --output "$OUTPUT"
