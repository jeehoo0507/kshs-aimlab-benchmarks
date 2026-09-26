#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TORCH_HOME="$PWD/.cache/torch"
if [[ -n "${REFERENCE_PYTHON:-}" ]]; then
  python_bin="$REFERENCE_PYTHON"
else
  python_bin="$PWD/.venv/bin/python"
  if [[ ! -x "$python_bin" ]] || ! "$python_bin" -c 'import torch, torchvision, numpy, PIL; assert torch.version.cuda is not None' >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
      export UV_CACHE_DIR="$PWD/.cache/uv"
      uv venv --python 3.11 --allow-existing .venv
      uv pip install --python "$python_bin" torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124
      uv pip install --python "$python_bin" -r requirements-reference.txt
    else
      python3 -m venv .venv
      "$python_bin" -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124
      "$python_bin" -m pip install -r requirements-reference.txt
    fi
  fi
fi

exec "$python_bin" -m reference.cli run "$@"
