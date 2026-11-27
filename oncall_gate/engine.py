"""Decision engines. All return the same shape:

    {"routing": {"model": str, "reason": str},
     "answers": {qid: {"type": "choice"|"score"|"noul", ...}}}

LayaEngine  - the real thing. Since laya 0.5 it wraps `laya.Agent()` (the "vega" engine: a frozen
              reader with a small decision engine on top). Every answer also carries Laya's own
              `route` ("auto" | "review"), `abstain`, `prediction_set` and `unbound` fields.
              LAYA_MODEL=classic keeps the older ModernBERT `laya.Router`.
DemoEngine  - keyword stand-in reading the same text so the app runs with no weights. NOT Laya.

Answer normalisation (so policy.py sees one shape whatever the backend):
  * severity["score"] is always a float on the 0-3 scale. The vega engine returns an integer
    level in `score` and the continuous value in `expected_score`; we keep the integer as
    `level` and put the continuous value in `score`.
  * answers without Laya's review fields (classic, demo, Ollama) simply lack `route`; the policy
    then falls back to the category margin.
"""
import math
import os
import re

from .questions import QUESTIONS

# LAYA_MODEL -> (backend, laya.Agent kwargs, human label)
MODELS = {
    "vega": ("vega", {}, "laya.Agent() - vega engine, 0.8B reader (default since laya 0.5)"),
    "vega-4b": ("vega", {"vegaml": "4b"}, "laya.Agent(vegaml='4b') - vega engine, 4B reader"),
    "classic": ("classic", {}, "laya.Router() - classic ModernBERT checkpoints"),
}
DEFAULT_MODEL = "vega"


def render_state(alert):
    """What Laya reads. Numbers stay in the prose; thresholds live in policy, not here."""
    return {"service": alert["service"],
            "tier": "tier-%d%s" % (alert["tier"], " (customer-facing critical path)" if alert["tier"] == 1 else ""),
            "source": alert["source"], "alert": alert["text"]}


def state_text(state):
    """The dict above as plain labelled lines (the vega reader takes text)."""
    return "\n".join("%s: %s" % (k, v) for k, v in state.items())


def normalise(res):
    """Make severity a 0-3 float on every backend (see module docstring). Mutates and returns res."""
    sev = res["answers"].get("severity")
    if sev and sev.get("expected_score") is not None:
        sev["level"] = sev.get("score")
        sev["score"] = float(sev["expected_score"])
    elif sev and sev.get("score") is not None:
        sev["score"] = float(sev["score"])
    return res


class LayaEngine:
    name = "laya"

    def __init__(self, model=None):
        os.environ.setdefault("USE_TF", "0")
        self.model_key = (model or os.environ.get("LAYA_MODEL", DEFAULT_MODEL)).lower()
        if self.model_key not in MODELS:
            raise ValueError("LAYA_MODEL=%r is not one of: %s" % (self.model_key, ", ".join(MODELS)))
        self.backend, kwargs, self.label = MODELS[self.model_key]
        device = os.environ.get("LAYA_DEVICE") or None
        if self.backend == "vega":
            import laya  # lazy: heavy (torch)
            self.model = laya.Agent(device=device, **kwargs)
        else:
            from laya import Router
            self.model = Router(device=device)

    def predict(self, state):
        if self.backend == "vega":
            res = self.model.predict(state_text(state), QUESTIONS)
            res.setdefault("routing", {"model": self.model_key, "reason": self.label})
        else:
            res = self.model.predict(state, QUESTIONS)
        return normalise(res)

    def describe(self):
        """Where the engine runs and what it loaded; shown on the Architecture tab."""
        m = self.model
        info = {"backend": self.backend, "model": self.model_key, "model_label": self.label}
        dev = getattr(m, "device", None)
        info["device"] = str(dev() if callable(dev) else dev) if dev is not None else None
        if self.backend == "vega":
            info["checkpoints_loaded"] = [str(getattr(m, "model_id", "vega"))]
            info["context_tokens"] = getattr(m, "context_tokens", None)
        else:
            loaded = getattr(m, "loaded", None)
            info["checkpoints_loaded"] = list(loaded) if loaded is not None else None
        return info


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

    def describe(self):
        return {"backend": "demo", "model": "demo", "model_label": "keyword stand-in (NOT Laya)"}


def build_engine(mode=None):
    """mode: laya | demo | auto (default: env LAYA_ENGINE or auto). Returns (engine, note)."""
    mode = (mode or os.environ.get("LAYA_ENGINE", "auto")).lower()
    if mode == "demo":
        return DemoEngine(), "Demo mode (keyword heuristic) - not real Laya output."
    try:
        eng = LayaEngine()
        eng.predict({"alert": "warm-up", "service": "x", "tier": "tier-3", "source": "x"})  # forces load
        return eng, "Laya engine ready (%s)." % eng.label
    except Exception as exc:  # noqa: BLE001
        if mode == "laya":
            raise
        return DemoEngine(), "Laya weights unavailable (%s: %s) - running DEMO stand-in." % (
            type(exc).__name__, str(exc)[:120])
