#!/usr/bin/env bash
cd "$(dirname "$0")"
# LAYA_ENGINE=laya  -> fail if weights unavailable | demo -> heuristic stand-in | auto (default)
exec .venv/bin/python app.py
