"""On-Call Gate - decide whether an application alert should page the on-call engineer.
Built on Laya (System-1 decision engine): one forward pass answers typed questions about each alert."""
import inspect
import os
import platform
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from oncall_gate import policy
from oncall_gate.alerts import ALERTS, BY_ID
from oncall_gate import engine as engine_mod
from oncall_gate import ollama_engine
from oncall_gate.compare import CompareJob
from oncall_gate.engine import build_engine, render_state
from oncall_gate.explain import review_fields
from oncall_gate.questions import QUESTIONS

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder="static")


def loc(obj):
    """file:line of a function/method, so the UI can point you at the exact code (Cmd+O in the IDE)."""
    try:
        f = inspect.getsourcefile(obj)
        line = inspect.getsourcelines(obj)[1]
        return "%s:%d" % (os.path.relpath(f, ROOT) if f.startswith(ROOT) else "laya library: " + os.path.basename(f), line)
    except Exception:  # noqa: BLE001
        return "n/a"


def answer_text(qid, a):
    if a["type"] == "choice":
        top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        return "%s (%s)" % (a["choice"], ", ".join("%s %.0f%%" % (k, v * 100) for k, v in top)), a["probabilities"][a["choice"]]
    if a["type"] == "score":
        return "%.2f of %d" % (a["score"], len(a["legend"]) - 1), a["score"] / (len(a["legend"]) - 1)
    return "%.0f%% yes" % (a["noul"] * 100), a["noul"]
ENGINE = None
ENGINE_LOCK = threading.Lock()          # one prediction at a time on the primary engine
JOB = CompareJob()
STATUS = {"state": "idle", "note": "", "error": None, "started": None, "finished": None}
STATE = policy.OnCallState()


def _load():
    """Load the decision engine. Runs in a background thread so the web server starts at once,
    even while Laya downloads its weights (several GB) on the first run."""
    global ENGINE
    STATUS.update(state="loading", started=time.time(), error=None)
    print("[oncall-gate] loading engine (first run downloads model weights; this can take minutes)...", flush=True)
    try:
        ENGINE, note = build_engine()
        STATUS.update(state="ready", note=note, finished=time.time())
        print("[oncall-gate] READY:", note, flush=True)
    except Exception as exc:  # noqa: BLE001  (only reachable with LAYA_ENGINE=laya)
        STATUS.update(state="failed", error="%s: %s" % (type(exc).__name__, exc), finished=time.time())
        print("[oncall-gate] FAILED:", STATUS["error"], flush=True)


def start_loader():
    if STATUS["state"] != "idle":
        return
    STATUS["state"] = "loading"
    if os.environ.get("LAYA_ENGINE", "auto").lower() == "demo":
        _load()                                    # instant
    else:
        threading.Thread(target=_load, daemon=True, name="engine-loader").start()


def engine():
    """The ready engine, or None while loading / failed."""
    start_loader()
    return ENGINE


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    engine()
    elapsed = round((STATUS["finished"] or time.time()) - (STATUS["started"] or time.time()), 1)
    return jsonify(state=STATUS["state"], engine=ENGINE.name if ENGINE else None, note=STATUS["note"],
                   error=STATUS["error"], elapsed_s=elapsed)


@app.get("/api/engine")
def engine_info():
    """Where is the decision engine actually running? Used by the Architecture tab."""
    e = engine()
    if e is None:
        return jsonify(state=STATUS["state"], engine=None, local=False, error=STATUS["error"],
                       note="Engine is still loading. First run downloads the Laya weights; watch the console.")
    info = {"state": STATUS["state"], "engine": e.name, "note": STATUS["note"], "local": e.name == "laya",
            "python": platform.python_version(), "platform": platform.platform(terse=True)}
    info.update(e.describe())
    if e.name == "laya":
        try:
            import laya, torch
            info["laya"], info["torch"] = laya.__version__, torch.__version__
        except Exception:  # noqa: BLE001
            pass
        info["weights_cache"] = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        info["hf_offline"] = os.environ.get("HF_HUB_OFFLINE", "0")
    return jsonify(info)


@app.get("/api/alerts")
def alerts():
    return jsonify([{k: v for k, v in a.items() if k != "truth"} for a in ALERTS])


@app.post("/api/reset")
def reset():
    STATE.reset()
    return jsonify(ok=True)


@app.post("/api/alert")
def alert_endpoint():
    if engine() is None:
        return jsonify(error="engine %s" % STATUS["state"], detail=STATUS["error"]), 503
    body = request.get_json(force=True)
    if body.get("id") in BY_ID:
        alert = dict(BY_ID[body["id"]])
    else:  # custom injected alert
        text = (body.get("text") or "").strip()
        if not text:
            return jsonify(error="id or text required"), 400
        alert = {"id": "X%d" % (len(STATE.events) + 1), "t": STATE.clock + 1,
                 "service": body.get("service") or "custom-svc", "tier": int(body.get("tier", 1)),
                 "source": "custom", "text": text, "truth": None,
                 "change_window": bool(body.get("change_window"))}
    t0 = time.perf_counter()
    with ENGINE_LOCK:
        res = ENGINE.predict(render_state(alert))
    ms = (time.perf_counter() - t0) * 1000
    ans = res["answers"]
    base = policy.decide(ans, alert)
    unsure, unsure_why = policy.review_signal(ans)
    action, note, inc, corr = STATE.apply(base, alert)
    state_sent = render_state(alert)
    laya_cls = type(ENGINE).__name__
    laya_fn = ENGINE.predict
    laya_rows = []
    for qid, q in QUESTIONS.items():
        txt, frac = answer_text(qid, ans[qid])
        laya_rows.append({"id": qid, "type": q["type"], "question": q["instructions"], "answer": txt,
                          "fraction": round(frac, 3), **review_fields(ans[qid])})
    pager = STATE.pages[-1]["message"] if action == "PAGE" else None
    path = {"steps": [
        {"n": 1, "title": "Alert received", "where": "app.py > alert_endpoint()", "loc": loc(alert_endpoint),
         "summary": "The browser POSTed the alert. Service, tier and text are read from the request.",
         "data": {"id": alert["id"], "service": alert["service"], "tier": alert["tier"], "text": alert["text"]}},
        {"n": 2, "title": "Alert rendered as text for the model", "where": "oncall_gate/engine.py > render_state()",
         "loc": loc(render_state), "summary": "Laya reads text only, so the alert is packed into this small dictionary.",
         "data": state_sent},
        {"n": 3, "title": "Laya judges the alert (one forward pass, 6 answers)",
         "where": "%s.predict() > %s" % (laya_cls, ENGINE.describe()["model_label"] if ENGINE.name == "laya" else "keyword stand-in (NOT Laya)"),
         "loc": loc(laya_fn), "summary": "Each question below is answered with a probability or level. No text is generated. Took %.1f ms.%s" % (
             ms, " %s." % unsure_why if ENGINE.name == "laya" else ""),
         "rows": laya_rows},
        {"n": 4, "title": "Policy rules applied (first matching rule wins)", "where": "oncall_gate/policy.py > decide()",
         "loc": loc(policy.decide), "summary": base["narrative"]["headline"], "table": base["table"],
         "rejected": base["narrative"]["earlier_rules_rejected"]},
        {"n": 5, "title": "Correlated with open incidents", "where": "oncall_gate/policy.py > OnCallState.apply()",
         "loc": loc(policy.OnCallState.apply), "summary": "Policy proposed %s; after correlation the final decision is %s." % (base["base_action"], action),
         "steps": corr["steps"]},
        {"n": 6, "title": "Final decision returned to the UI", "where": "app.py > alert_endpoint()", "loc": loc(alert_endpoint),
         "summary": "Final: %s%s." % (action, " (" + inc + ")" if inc else ""), "pager": pager}]}
    ev = {"id": alert["id"], "t": alert["t"], "service": alert["service"], "tier": alert["tier"],
          "text": alert["text"], "truth": alert["truth"], "engine": ENGINE.name,
          "state_sent": state_sent,
          "latency_ms": round(ms, 1), "routing": res.get("routing", {}),
          "answers": {"category": ans["category"]["choice"],
                      "category_p": round(ans["category"]["probabilities"][ans["category"]["choice"]], 3),
                      **{k: round(ans[k]["noul"], 3) for k in
                         ("customer_impact", "self_recovering", "sustained", "business_risk")},
                      "severity": round(ans["severity"]["score"], 2)},
          "review": {"needs_review": unsure, "why": unsure_why,
                     "flagged": [q for q in policy.REVIEW_QS if ans[q].get("route") == "review"]},
          "decision": {"action": action, "base_action": base["base_action"], "rule": base["rule"],
                       "reasons": base["reasons"] + ([note] if note else []), "incident": inc},
          "path": path}
    ev["correct"] = policy.correct(action, alert["truth"])
    STATE.events.append(ev)
    return jsonify(ev)


@app.get("/api/policy")
def policy_table():
    return jsonify(policy.rules_doc())


@app.get("/api/ollama/models")
def ollama_models():
    url = request.args.get("url") or ollama_engine.DEFAULT_URL
    try:
        return jsonify(ok=True, url=url, models=ollama_engine.list_models(url), default=ollama_engine.DEFAULT_MODEL)
    except ollama_engine.OllamaError as exc:
        return jsonify(ok=False, url=url, error=str(exc)), 200


@app.post("/api/compare/start")
def compare_start():
    body = request.get_json(force=True) or {}
    primary = engine()
    if primary is None:
        return jsonify(error="primary engine is still %s" % STATUS["state"]), 503
    url, model = body.get("url") or ollama_engine.DEFAULT_URL, body.get("model") or ollama_engine.DEFAULT_MODEL
    first = ("Laya" if primary.name == "laya" else "Demo stand-in (NOT Laya)", primary, primary.name != "laya")
    second = ("Ollama " + model, ollama_engine.OllamaEngine(model, url), False)
    if not JOB.start([first, second], lock=ENGINE_LOCK):
        return jsonify(error="a comparison is already running"), 409
    return jsonify(ok=True)


@app.get("/api/compare/status")
def compare_status():
    return jsonify(status=JOB.status, result=JOB.result)


@app.get("/api/scorecard")
def scorecard():
    sc = policy.scorecard(STATE.events)
    sc["engine"] = ENGINE.name if ENGINE else None
    sc["pager_log"] = STATE.pages
    sc["incidents"] = list(STATE.incidents.values())
    return jsonify(sc)


if __name__ == "__main__":
    start_loader()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8080)), threaded=True)
