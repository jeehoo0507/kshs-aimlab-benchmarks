#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/MaskedKD"
REF="96d052da7346441e2425876a9ad37ff8c87fe383"

if [[ -e "$DEST" ]]; then
  [[ -d "$DEST/.git" ]] || { echo "Existing path is not a Git checkout: $DEST" >&2; exit 1; }
  CURRENT="$(git -C "$DEST" rev-parse HEAD)"
  [[ "$CURRENT" == "$REF" ]] || { echo "Existing checkout is at $CURRENT, expected $REF" >&2; exit 1; }
  echo "Verified MaskedKD at $REF"
  exit 0
fi

mkdir -p "$ROOT/external"
git clone https://github.com/effl-lab/MaskedKD.git "$DEST"
git -C "$DEST" checkout --detach "$REF"
echo "MaskedKD ready at $DEST ($REF)"
