#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -gt 2 ]]; then
  echo "Usage: bash datasets/imagenet/setup.sh [archive-directory] [output-directory]" >&2
  exit 2
fi
SOURCE="${1:-data/imagenet/archives}"
OUTPUT="${2:-data/imagenet}"

if [[ ! -f "$OUTPUT/manifest.json" ]]; then
  for filename in ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar ILSVRC2012_devkit_t12.tar.gz; do
    if [[ ! -f "$SOURCE/$filename" ]]; then
      echo "Missing $SOURCE/$filename" >&2
      echo "Download the official archives and place them in $SOURCE (see datasets/imagenet/README.md)." >&2
      exit 1
    fi
  done
fi

if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="$PWD/.cache/uv"
  uv venv --python 3.11 --allow-existing .venv-data
  uv pip install --python .venv-data/bin/python -r datasets/imagenet/requirements.txt
else
  python3 -m venv .venv-data
  .venv-data/bin/python -m pip install -r datasets/imagenet/requirements.txt
fi

.venv-data/bin/python datasets/imagenet/prepare.py --source "$SOURCE" --output "$OUTPUT"
