"""Ollama as a drop-in decision engine: answers the SAME six questions (questions.py) as Laya,
but by generating JSON text with a local LLM. Same output shape as Laya, so the SAME policy runs on it.

Differences worth knowing (they show up in the comparison):
  * An LLM writes tokens (JSON); Laya does one forward pass and writes none.
  * Numbers are the LLM's *stated* estimates, not calibrated model probabilities.
  * category is a single label, so we report it as one-hot (margin 1.0); no Laya `route` field,
    so rule R9 (model unsure) falls back to the margin test and never fires.
"""
import json
import os
import urllib.error
import urllib.request

from .questions import QUESTIONS

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
NOULS = ("customer_impact", "self_recovering", "sustained", "business_risk")


class OllamaError(RuntimeError):
    pass


def _http(url, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"},
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise OllamaError("HTTP %s from Ollama: %s" % (e.code, e.read().decode()[:300]))
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError("Cannot reach Ollama at %s (%s). Start it with `ollama serve` (or open the Ollama app)." % (url, e))


def list_models(url=DEFAULT_URL):
    return [m["name"] for m in _http(url.rstrip("/") + "/api/tags", timeout=5).get("models", [])]


def _schema():
    num = {"type": "number"}
    props = {"category": {"type": "string", "enum": list(QUESTIONS["category"]["criteria"])}}
    props.update({k: num for k in NOULS})
    props["severity"] = num
    return {"type": "object", "properties": props, "required": list(props)}


def _prompt(state):
    q = QUESTIONS
    cats = "\n".join("  - %s: %s" % kv for kv in q["category"]["criteria"].items())
    sev = ", ".join("%d=%s" % (i, c) for i, c in enumerate(q["severity"]["criteria"]))
    return (
        "You triage application monitoring alerts. Read the alert and answer ALL questions as JSON.\n\n"
        "ALERT\n  service: %(service)s\n  tier: %(tier)s\n  source: %(source)s\n  text: %(alert)s\n\n"
        "QUESTIONS\n"
        "1. category - %(c)s Choose exactly one:\n%(cats)s\n"
        "2. customer_impact - %(ci)s Probability 0 to 1.\n"
        "3. self_recovering - %(sr)s Probability 0 to 1.\n"
        "4. sustained - %(su)s Probability 0 to 1.\n"
        "5. business_risk - %(br)s Probability 0 to 1.\n"
        "6. severity - %(sv)s Number from 0 to 3 (%(sev)s).\n\n"
        "Reply with JSON only."
    ) % dict(state, c=q["category"]["instructions"], cats=cats, ci=q["customer_impact"]["instructions"],
             sr=q["self_recovering"]["instructions"], su=q["sustained"]["instructions"],
             br=q["business_risk"]["instructions"], sv=q["severity"]["instructions"], sev=sev)


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        raise OllamaError("non-numeric value in model output: %r" % (v,))


class OllamaEngine:
    name = "ollama"

    def __init__(self, model=None, url=None):
        self.model, self.url = model or DEFAULT_MODEL, (url or DEFAULT_URL).rstrip("/")
        self.last_stats = {}

    def predict(self, state):
        payload = {"model": self.model, "stream": False, "keep_alive": "15m", "format": _schema(),
                   "messages": [{"role": "system", "content": "You are a precise SRE alert triage assistant. Output JSON only."},
                                {"role": "user", "content": _prompt(state)}],
                   "options": {"temperature": 0, "seed": 7, "num_predict": 220}}
        last = None
        for _ in range(2):                                   # one retry on malformed output
            r = _http(self.url + "/api/chat", payload)
            self.last_stats = {"total_ms": r.get("total_duration", 0) / 1e6, "load_ms": r.get("load_duration", 0) / 1e6,
                               "prompt_tokens": r.get("prompt_eval_count", 0), "output_tokens": r.get("eval_count", 0),
                               "eval_ms": r.get("eval_duration", 0) / 1e6}
            try:
                d = json.loads(r["message"]["content"])
                cat = d["category"]
                if cat not in QUESTIONS["category"]["criteria"]:
                    raise OllamaError("unknown category %r" % (cat,))
                probs = {k: (1.0 if k == cat else 0.0) for k in QUESTIONS["category"]["criteria"]}
                ans = {"category": {"type": "choice", "choice": cat, "probabilities": probs, "confidence": 1.0,
                                    "answer_confidence": 1.0}}
                for k in NOULS:
                    ans[k] = {"type": "noul", "noul": _clamp(d[k], 0, 1)}
                ans["severity"] = {"type": "score", "score": _clamp(d["severity"], 0, 3),
                                   "legend": QUESTIONS["severity"]["criteria"]}
                return {"routing": {"model": self.model, "reason": "local Ollama LLM, JSON output"}, "answers": ans}
            except (KeyError, ValueError, TypeError, OllamaError) as e:
                last = e
        raise OllamaError("malformed model output after retry: %s" % last)
