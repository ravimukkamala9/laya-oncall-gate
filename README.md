# On-Call Gate — should this alert wake the on-call engineer?

A small app that streams application alerts (database, memory, latency, timeouts, transaction failures,
IBM MQ, maintenance, noise) through **Laya** (open System-1 decision model, Apache-2.0). One forward
pass answers six typed questions per alert; an auditable policy turns them into
**PAGE / MERGED / WATCH / TICKET / SUPPRESS**. Same-service alerts within 10 minutes merge into one incident.
A scorecard compares decisions to hidden ground truth: pages avoided, accuracy, **missed pages**, false pages.

| Laya primitive | Question |
|---|---|
| `choice` | `category`: database, transaction_failure, memory, latency, timeout, messaging, infra_other |
| `noul` (yes/no probability) | `customer_impact`, `self_recovering`, `sustained`, `business_risk` |
| `score` | `severity` 0–3 |

## Connect to Laya locally (step by step)
1. **Install** (once): `./setup.sh`, or `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
2. **Check Laya** (once): run `python scripts/check_laya.py` (IntelliJ: new Python run config, script `scripts/check_laya.py`,
   env `USE_TF=0`). It tests PyTorch, Laya, Hugging Face access, downloads the weights (~1.3 GB, first time only, a few
   minutes) and sends a real alert through Laya. Every step prints OK/FAIL with the exact error.
3. **Start the app** with env `LAYA_ENGINE=laya;USE_TF=0` (fails loudly instead of silently using the demo engine).
   The server starts immediately; a blue banner shows *Loading Laya...* and **Replay stays disabled** until the model is ready.
   When ready the banner turns green: *Real Laya is running locally*.
4. Open the **Architecture** tab: the "Is Laya running locally?" panel shows the engine, device (mps/cpu), loaded checkpoints.
5. Prove it is offline: stop, restart with `HF_HUB_OFFLINE=1` added. It should still work (weights are cached in `~/.cache/huggingface`).

## Compare Laya with a local Ollama model
1. Install Ollama (https://ollama.com, or `brew install ollama`), start it (`ollama serve`, or open the Ollama app) and pull a model: `ollama pull llama3.2`.
   Check: `curl localhost:11434/api/tags` lists your models.
2. Start the app with **real Laya** (`LAYA_ENGINE=laya`) so the left column is Laya, not the stand-in.
3. Open **Laya vs Ollama** → *Check Ollama* → pick a model → *Run comparison*. Both engines answer the same six questions for the 24 alerts and feed
   the same rule table (`policy.py`). You get latency (mean/p95/total), throughput, accuracy vs expected, missed/false pages, tokens generated,
   errors, decision agreement and an alert-by-alert table. The landing page shows a one-line summary.
4. Settings: `OLLAMA_URL` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default `llama3.2`) env vars set the defaults. Code: `oncall_gate/ollama_engine.py`, `oncall_gate/compare.py`.
   Tests use a fake Ollama (`tests/mock_ollama.py`), not a real LLM.
4. In **Alert by alert**, click any row (or its *trace* link) to see, for Laya and Ollama side by side: the six answers, every rule R1..R11 evaluated with actual value vs threshold, the correlation steps, and a plain-language "why" (for a wrong result it names the closest PAGE rule and the condition that blocked it).

## Tabs
- **Alert replay** – stream the alerts and watch decisions, pager and scorecard.
- On every alert, **Show decision path** opens the full trace: which class/function did each step (with `file:line`),
  Laya's six answers, the **decision table** (every rule, every condition with its actual value, ✓/✗, winner highlighted)
  and the incident-correlation steps. Custom alerts open their path automatically.
- **Architecture** – diagram of the pipeline, a step-by-step explanation, a live *"Is Laya running locally?"* panel
  (engine, device, checkpoints, weights cache) and a trace of the last alert (text sent to Laya → answers → rule → correlation).

## Run
```bash
./setup.sh   # once: venv + pip install laya + download weights from Hugging Face
./run.sh     # http://127.0.0.1:8080  -> press Replay
```
`LAYA_ENGINE=laya` fails loudly if weights are missing; `demo` uses a keyword stand-in (UI shows a DEMO banner;
its output is **not** Laya); default `auto` tries Laya, then falls back.

## Layout
- `oncall_gate/questions.py` – decision schema (what we ask Laya)
- `oncall_gate/engine.py` – `LayaEngine` (real `Router`) and `DemoEngine` (stand-in)
- `oncall_gate/policy.py` – thresholds, rules, incident correlation, simulated pager, scorecard
- `oncall_gate/alerts.py` – the 24 scripted alerts + ground truth
- `app.py` – Flask API (`/api/alert`, `/api/alerts`, `/api/scorecard`, `/api/reset`, `/api/health`) and UI in `static/`
- `tests/` – pytest (demo engine)

## Run in an IDE
### VS Code (configs in `.vscode/`)
1. `File > Open Folder…` → this folder; install the **Python** extension.
2. Terminal: `./setup.sh` (skip for demo-only).
3. `Cmd+Shift+P` → *Python: Select Interpreter* → `./.venv/bin/python`.
4. Run and Debug (`Cmd+Shift+D`) → choose *On-Call Gate + Laya (auto)*, *real Laya only*, *demo*, or *Pytest* → ▶.
5. Open http://127.0.0.1:8080. Put breakpoints in `oncall_gate/policy.py` (`decide`) or `app.py` (`alert_endpoint`).

### PyCharm
Open the folder, set interpreter to `.venv/bin/python`, add a Python run config: script `app.py`,
env `LAYA_ENGINE=auto;USE_TF=0;PORT=8080`. For tests add a pytest config on `tests` with `LAYA_ENGINE=demo`.

### Troubleshooting
- **Process seems stopped / nothing happens on first run** → it is downloading weights. Watch the console and the blue banner;
  run `scripts/check_laya.py` to see progress and any error.
- `ModuleNotFoundError` → wrong interpreter; select the project's `.venv` (see above).
- `403` / `ProxyError` / `ConnectionError` talking to huggingface.co → corporate proxy or no internet. Set `HTTPS_PROXY`, or download
  the weights on another network and copy `~/.cache/huggingface` over; then run with `HF_HUB_OFFLINE=1`.
- Red banner *Laya failed to load* → the error text is shown; fix it, restart (only happens with `LAYA_ENGINE=laya`).
- Orange **DEMO MODE** banner → weights not loaded (check access to huggingface.co; rerun `./setup.sh`).
- Port busy → `PORT=8081`. First real-Laya call is slow (model load); later calls take tens of ms.
- Apple Silicon uses MPS automatically; `LAYA_DEVICE=cpu` forces CPU.

## Notes
Probabilities are not calibrated for your alerts. Before trusting thresholds, replay labeled historical alerts
and tune `policy.py`. A model is a triage aid; keep humans in the loop for tier-1 decisions.
