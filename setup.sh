#!/usr/bin/env bash
# One-time setup: venv, install Laya, download the real model weights from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "Downloading Laya checkpoint (English ~1.3 GB; cached under ~/.cache/huggingface) ..."
.venv/bin/python - <<'PY'
import os
os.environ.setdefault("USE_TF", "0")
from laya import Router
r = Router()
r.predict({"alert": "warm-up"}, {"q": {"type": "noul", "instructions": "Is this a test?"}})
print("loaded checkpoints:", r.loaded)
PY
echo "Done. Start the app with ./run.sh  (then open http://127.0.0.1:8080)"
