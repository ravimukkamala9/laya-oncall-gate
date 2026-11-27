#!/usr/bin/env bash
# Start the app on http://127.0.0.1:${PORT:-8080}
#   LAYA_ENGINE=auto (default) try Laya, fall back to demo | laya: fail loudly | demo: no model
#   LAYA_MODEL=vega (default) | vega-4b | classic
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv found. Run ./setup.sh first (or ./setup.sh --no-model for demo mode)." >&2
  exit 1
fi
export USE_TF="${USE_TF:-0}"
exec .venv/bin/python app.py
