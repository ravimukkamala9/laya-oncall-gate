"""Decision engines. Both return the same shape as laya.Router.predict():

    {"routing": {"model": str, "reason": str},
     "answers": {qid: {"type": "choice"|"score"|"noul", ...}}}

LayaEngine  - the real thing (needs the model weights from Hugging Face).
DemoEngine  - keyword stand-in reading the same text so the app runs with no weights. NOT Laya.
"""
import math
import os
import re

from .questions import QUESTIONS


def render_state(alert):
    """The text Laya reads. Numbers stay in the prose; thresholds live in policy, not here."""
    return {"service": alert["service"],
            "tier": "tier-%d%s" % (alert["tier"], " (customer-facing critical path)" if alert["tier"] == 1 else ""),
            "source": alert["source"], "alert": alert["text"]}


class LayaEngine:
    name = "laya"

    def __init__(self):
        os.environ.setdefault("USE_TF", "0")
        from laya import Router  # lazy: heavy (torch)
        self.router = Router()

    def predict(self, state):
        return self.router.predict(state, QUESTIONS)


# ------------------------------------------------------------------------------- demo stand-in
_CAT = {
    "database": ["replication", "connection pool", "slow queries", "commits"],
    "transaction_failure": ["failed transactions", "transaction failure"],
    "memory": ["heap", "oom", "memory", "gc "],
    "latency": ["latency", "p99", "p95"],
    "timeout": ["timeout"],
    "messaging": ["queue", "channel", "dead-letter", "consumers"],
    "infra_other": ["cpu", "disk", "health check"],
}
_IMPACT = ["customers", "users", "sign-ins", "login", "checkout pages", "order processing"]
_RECOV = ["auto-recovered", "self-resolved", "recovered", "transient", "flap", "retry succeeded",
          "duplicate", "change window", "expected"]
_SUST = ["sustained", "rising", "worsening", "growing", "not recovering", "stuck"]
_RISK = ["failed transactions", "transaction commits", "writes are delayed", "order creation",
         "transaction failure", "data loss", "processing delayed"]


def _n(t, ws):
    return sum(1 for w in ws if w in t)


def _sig(x):
    return 1 / (1 + math.exp(-x))


class DemoEngine:
    name = "demo"

    def predict(self, state):
        t = state["alert"].lower()
        calm = "no customer impact" in t
        imp = 0.05 if calm else _sig(-2.0 + 2.6 * _n(t, _IMPACT))
        rec = _sig(-2.0 + 2.6 * _n(t, _RECOV))
        m = re.search(r"(\d+)\s*(?:minutes|mins|min)\b", t)
        sus = _sig(-1.5 + 1.2 * _n(t, _SUST) + (1.0 if m and int(m.group(1)) >= 4 else 0))
        risk = _sig(-2.0 + 2.2 * _n(t, _RISK))
        sev = max(0.0, min(3.0, 1.6 * imp + 0.9 * sus + 0.6 * risk - 1.4 * rec + 0.3))
        ex = {k: math.exp(1.5 * _n(t, v)) for k, v in _CAT.items()}
        z = sum(ex.values())
        probs = {k: v / z for k, v in ex.items()}
        top = max(probs, key=probs.get)
        o = sorted(probs.values(), reverse=True)
        return {"routing": {"model": "demo-heuristic", "reason": "keyword stand-in, not Laya"},
                "answers": {
                    "category": {"type": "choice", "choice": top, "probabilities": probs,
                                 "confidence": probs[top], "answer_confidence": probs[top] - o[1]},
                    "customer_impact": {"type": "noul", "noul": imp},
                    "self_recovering": {"type": "noul", "noul": rec},
                    "sustained": {"type": "noul", "noul": sus},
                    "business_risk": {"type": "noul", "noul": risk},
                    "severity": {"type": "score", "score": sev, "legend": QUESTIONS["severity"]["criteria"]}}}


def build_engine(mode=None):
    """mode: laya | demo | auto (default: env LAYA_ENGINE or auto). Returns (engine, note)."""
    mode = (mode or os.environ.get("LAYA_ENGINE", "auto")).lower()
    if mode == "demo":
        return DemoEngine(), "Demo mode (keyword heuristic) - not real Laya output."
    try:
        eng = LayaEngine()
        eng.predict({"alert": "warm-up", "service": "x", "tier": "tier-3", "source": "x"})  # forces load
        return eng, "Laya engine ready."
    except Exception as exc:  # noqa: BLE001
        if mode == "laya":
            raise
        return DemoEngine(), "Laya weights unavailable (%s: %s) - running DEMO stand-in." % (
            type(exc).__name__, str(exc)[:120])
