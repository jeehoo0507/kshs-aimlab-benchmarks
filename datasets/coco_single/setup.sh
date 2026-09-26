#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python3 -m venv .venv-data
.venv-data/bin/python -m pip install -r datasets/coco_single/requirements.txt
.venv-data/bin/python datasets/coco_single/prepare.py
