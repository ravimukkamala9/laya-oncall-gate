# On-Call Gate: should this alert wake the on-call engineer?

A small local web app that streams application alerts (database, memory, latency, timeouts, transaction
failures, IBM MQ, maintenance, noise) through **[Laya](https://pypi.org/project/laya/)**, an open
(Apache-2.0) decision model that runs on your own machine. One forward pass answers six typed questions per
alert. An auditable rule table then turns the answers into **PAGE / MERGED / WATCH / TICKET / SUPPRESS**.
Alerts on the same service within 10 minutes merge into one incident. A scorecard compares every decision
with hidden ground truth: pages avoided, accuracy, **missed pages** and false pages.

```
alert ──► render_state() ──► Laya (6 answers + its own route) ──► policy.py rules ──► incident correlation ──► decision
```

---

## Quick start

### 1. Prerequisites

| | |
|---|---|
| **Python 3.10 or newer** | `python3 --version`. Laya 0.5 needs 3.10+. On macOS: `brew install python@3.12` |
| **git** | to clone the repo |
| **Disk / RAM** | ~5 GB free (PyTorch ~1 GB, default model weights ~2 GB); 8 GB+ RAM recommended. Only needed for real Laya, not demo mode |
| **Internet, once** | to download the weights from Hugging Face. After that it runs offline |

Works on macOS (Apple Silicon uses the GPU via MPS), Linux and Windows (WSL). No GPU is required.

### 2. Clone

```bash
git clone https://github.com/ravimukkamala9/laya-oncall-gate.git
cd laya-oncall-gate
```

### 3a. Fastest: demo mode (about 1 minute, no model download)

Use this to see the UI and the rules working. The answers come from a keyword stand-in, **not Laya**.

```bash
./setup.sh --no-model          # creates .venv and installs dependencies
LAYA_ENGINE=demo ./run.sh      # open http://127.0.0.1:8080 and press Replay
```

### 3b. With real Laya (first run downloads ~2 GB of weights, about 5 minutes)

```bash
./setup.sh                     # venv + dependencies + downloads the weights and runs a self-check
./run.sh                       # open http://127.0.0.1:8080
```

`./setup.sh` ends by running `scripts/check_laya.py`, which prints `[ OK ]` or `[FAIL]` with the exact
error for each step: PyTorch, Laya, Hugging Face access, weights cached, model load, one real alert.
The last line should say **ALL GOOD**.

When the app starts, a blue banner says *Loading Laya...* and **Replay stays disabled** until the model is
ready. Then the banner turns green: *Real Laya is running locally*. After the first download, the model
loads in a few seconds.

### 4. Check it works

```bash
.venv/bin/python -m pytest -q tests      # 20 tests, run on the demo engine: no weights needed
```

In the browser:
1. **Alert replay**: press *Replay*. Watch decisions, the pager and the scorecard fill in.
   Click **Show decision path** on any alert for the full trace.
2. **Architecture**: the *Is Laya running locally?* panel shows engine, model, device and Laya/PyTorch versions.
3. **Prove it is offline**: stop the app and restart with `HF_HUB_OFFLINE=1 ./run.sh`. It still works.

<details>
<summary>Manual setup (without the scripts, or on Windows)</summary>

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt           # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python scripts/check_laya.py              # optional: download weights + self-check
LAYA_ENGINE=auto USE_TF=0 .venv/bin/python app.py   # Windows (PowerShell): $env:LAYA_ENGINE="auto"; .venv\Scripts\python app.py
```
</details>

---

## Configuration

Everything is set with environment variables, for example `LAYA_MODEL=vega-4b PORT=8081 ./run.sh`.

| Variable | Default | Values |
|---|---|---|
| `LAYA_ENGINE` | `auto` | `auto` tries Laya and falls back to demo (orange DEMO banner). `laya` fails loudly if Laya can't load. `demo` is the keyword stand-in, no download |
| `LAYA_MODEL` | `vega` | `vega`: `laya.Agent()`, the default engine since laya 0.5, with a 0.8B reader. `vega-4b`: the same engine with a 4B reader (slower, bigger download). `classic`: the older ModernBERT `laya.Router` |
| `LAYA_DEVICE` | auto | `cpu`, `mps` or `cuda` to force a device |
| `PORT` | `8080` | web server port |
| `HF_HUB_OFFLINE` | `0` | `1` = never contact Hugging Face (weights must already be cached) |
| `HF_HOME` | `~/.cache/huggingface` | where the weights are cached |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.2` | defaults for the Laya vs Ollama tab |

---

## What Laya is asked

| Laya primitive | Question |
|---|---|
| `choice` | `category`: database, transaction_failure, memory, latency, timeout, messaging, infra_other |
| `noul` (yes/no probability) | `customer_impact`, `self_recovering`, `sustained`, `business_risk` |
| `score` | `severity` 0–3 |

The schema lives in `oncall_gate/questions.py`; the rules in `oncall_gate/policy.py`.

### New with laya 0.5: Laya says when it is unsure

On the default engine (`LAYA_MODEL=vega`), every answer also carries Laya's own uncertainty fields:

| Field | Meaning |
|---|---|
| `route` | `auto` or `review`. `review` when Laya abstains or its prediction set holds more than one option |
| `prediction_set` | every option the answer could still be (conformal set, 90% coverage) |
| `abstain` | Laya declines to commit at its calibrated risk level |
| `unbound` | the input looks unlike anything it knows. Shown, but not used for routing |

Laya's guidance is to **gate on `route`, not on a raw confidence number**, so rule **R9** ("model unsure on
tier-1: lean to page") now fires when Laya routes a key answer (category, impact, sustained, business risk,
severity) to `review`. Engines without these fields (classic Router, demo, Ollama) fall back to the old test,
a category margin below 0.10. The decision path shows a **Laya review** column with the route, the
prediction set and any abstain or unbound flags for each answer.

On the new engine, severity is returned as an integer level plus a continuous `expected_score`. The app
feeds the continuous value to the policy (so `severity >= 1.5` still means something) and keeps the integer
as `level`.

---

## What to expect (measured)

Measured on an Apple Silicon Mac (MPS), laya 0.5.5, `LAYA_MODEL=vega`, 24 scripted alerts, offline:

| | Real Laya (vega, zero-shot) | Demo stand-in |
|---|---|---|
| Latency per alert | 1.1–1.6 s | < 1 ms |
| Accuracy vs ground truth | 67% (16/24) | ≥ 90% (it was written for these alerts) |
| Pages sent / missed / false | 5 / 1 (L1) / 2 (D1, M1) | 0 missed, 0 false |

Read this honestly: the demo stand-in's keywords were written for these 24 alerts, so its score says nothing
about real alerts. Real Laya is answering cold, with no examples from your domain. Most of its misses
are noise alerts (N1–N5) that it ticketed instead of suppressing. Laya marks `category` and `severity` as
`review` on almost every alert, which is its way of saying it hasn't been calibrated for this domain.

On the same Laya answers, R9 on Laya's `route` scored 67% with 1 missed page; the old margin test scored
62.5% with 2 missed pages (M2 and L1). To do better, tune the thresholds in `policy.py` on your own labelled
alerts, or fit Laya on them (see *Notes*).

The classic encoder (`LAYA_MODEL=classic`) is much faster per call (tens of ms) but doesn't return
`route` / `prediction_set`.

---

## Compare Laya with a local Ollama model

1. Install Ollama (https://ollama.com, or `brew install ollama`), start it (`ollama serve`) and pull a model:
   `ollama pull llama3.2`. Check with `curl localhost:11434/api/tags`.
2. Start the app with real Laya (`LAYA_ENGINE=laya ./run.sh`) so the left column is Laya, not the stand-in.
3. Open **Laya vs Ollama** → *Check Ollama* → pick a model → *Run comparison*. Both engines answer the same
   six questions for the 24 alerts and feed the same rule table. You get latency (mean/p95/total),
   throughput, accuracy, missed and false pages, tokens generated, errors, decision agreement and an
   alert-by-alert table.
4. Click any row in **Alert by alert** to see both engines side by side: the six answers, every rule R1..R11
   with actual value vs threshold, the correlation steps and a plain-language "why".

The tests use a fake Ollama (`tests/mock_ollama.py`), not a real LLM.

---

## Run in an IDE

### VS Code (configs in `.vscode/`)
1. `File > Open Folder…` → this folder; install the **Python** extension.
2. Terminal: `./setup.sh` (or `./setup.sh --no-model` for demo only).
3. `Cmd+Shift+P` → *Python: Select Interpreter* → `./.venv/bin/python`.
4. Run and Debug (`Cmd+Shift+D`) → *1. Check Laya*, *On-Call Gate + Laya (auto)*, *real Laya only*,
   *classic Laya Router*, *demo* or *Pytest* → ▶.
5. Set breakpoints in `oncall_gate/policy.py` (`decide`) or `app.py` (`alert_endpoint`).

### PyCharm / IntelliJ
Set the interpreter to `.venv/bin/python` and add a Python run config: script `app.py`,
env `LAYA_ENGINE=auto;LAYA_MODEL=vega;USE_TF=0;PORT=8080`. For tests add a pytest config on `tests` with
`LAYA_ENGINE=demo`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python 3.10+ is required` from `setup.sh` | Install a newer Python, then `PYTHON=python3.12 ./setup.sh` |
| `No .venv found` from `run.sh` | Run `./setup.sh` first |
| Nothing seems to happen on first run | It is downloading weights. Watch the console and the blue banner, or run `.venv/bin/python scripts/check_laya.py` to see each step |
| `ModuleNotFoundError` | Wrong interpreter. Use `.venv/bin/python` (see the IDE section) |
| `403` / `ProxyError` / `ConnectionError` to huggingface.co | Corporate proxy or no internet. Set `HTTPS_PROXY`, or download on another network, copy `~/.cache/huggingface` over and run with `HF_HUB_OFFLINE=1` |
| Orange **DEMO MODE** banner | Laya didn't load and `auto` fell back. The banner shows why. Rerun `./setup.sh`, or start with `LAYA_ENGINE=laya` to fail loudly |
| Red *Laya failed to load* banner | Only with `LAYA_ENGINE=laya`. The error is shown; fix it and restart |
| Slow or out of memory on an older laptop | Use the default `vega` (not `vega-4b`), try `LAYA_MODEL=classic` (smaller encoder), or force `LAYA_DEVICE=cpu` |
| Port busy | `PORT=8081 ./run.sh` |
| Upgrading an old checkout (laya 0.3/0.4) | `.venv/bin/pip install -U -r requirements.txt`. If your venv is Python 3.9, delete `.venv` and rerun `./setup.sh` |

---

## Project layout

```
app.py                     Flask API + serves the UI in static/
oncall_gate/questions.py   the six questions asked of Laya
oncall_gate/engine.py      LayaEngine (laya.Agent / classic Router), DemoEngine, answer normalisation
oncall_gate/policy.py      thresholds, rule table R1..R11, review signal, correlation, pager, scorecard
oncall_gate/alerts.py      24 scripted alerts + ground truth
oncall_gate/explain.py     plain-language trace for each decision
oncall_gate/compare.py     Laya vs Ollama runner
oncall_gate/ollama_engine.py  Ollama as a drop-in engine
scripts/check_laya.py      step-by-step diagnostic: .venv/bin/python scripts/check_laya.py [vega|vega-4b|classic]
tests/                     pytest (demo engine + laya 0.5 answer shapes; no weights needed)
```

API: `GET /api/health`, `/api/engine`, `/api/alerts`, `/api/policy`, `/api/scorecard`;
`POST /api/alert`, `/api/reset`; `GET /api/ollama/models`; `POST /api/compare/start`, `GET /api/compare/status`.

## Notes

Probabilities are not calibrated for *your* alerts. Before trusting thresholds, replay labelled historical
alerts and tune `policy.py`. Laya 0.5 can also fit and calibrate on your own labels (`agent.fit(...)`,
at least about 35 labelled rows, a few hundred for a trustworthy `route`); the 24 demo alerts are too few
for that. A model is a triage aid: keep humans in the loop for tier-1 decisions.

Continuous learning in Laya (`laya.Learner`, `laya.Escalator`) is a separate, **noncommercial-only**
package (`vegaml-learn`) and is not used here. Everything this app uses is Apache-2.0.
