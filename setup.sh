#!/usr/bin/env bash
# One-time setup: venv, install Laya, download the model weights from Hugging Face.
#   ./setup.sh            -> venv + deps + download weights for LAYA_MODEL (default: vega)
#   ./setup.sh --no-model -> venv + deps only (enough for demo mode and the tests)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10+ is required (found: $("$PY" --version 2>&1)). Set PYTHON=/path/to/python3.12 and rerun." >&2
  exit 1
fi

# Recreate .venv if it was built with an older Python (venv keeps an existing interpreter link).
if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Existing .venv uses $(.venv/bin/python --version 2>&1); recreating it with $("$PY" --version 2>&1)."
  "$PY" -m venv --clear .venv
else
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "Installed: $(.venv/bin/python -c 'import laya; print("laya", laya.__version__)')"

if [[ "${1:-}" == "--no-model" ]]; then
  echo "Skipped model download. Start in demo mode with: LAYA_ENGINE=demo ./run.sh"
  exit 0
fi

echo "Downloading Laya weights for LAYA_MODEL=${LAYA_MODEL:-vega} (several GB, cached under ~/.cache/huggingface) ..."
USE_TF=0 .venv/bin/python scripts/check_laya.py
echo "Done. Start the app with ./run.sh  (then open http://127.0.0.1:${PORT:-8080})"
