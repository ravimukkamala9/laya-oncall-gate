"""Run the SAME alerts through two engines and the SAME policy; measure speed and quality."""
import statistics
import threading
import time

from . import explain, policy
from .alerts import ALERTS
from .engine import render_state


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else 0


def run_engine(label, engine, progress, lock=None, is_stand_in=False):
    """Returns {'label','summary','events'}. progress(done) is called after every alert."""
    state, events, lat, errors, out_tok = policy.OnCallState(), [], [], 0, 0
    # warm-up: loads the model / compiles kernels; excluded from latency stats, reported as load time
    t0 = time.perf_counter()
    load_note = None
    try:
        engine.predict(render_state(ALERTS[0]))
    except Exception as exc:  # noqa: BLE001
        load_note = "%s: %s" % (type(exc).__name__, exc)
    warm_s = time.perf_counter() - t0
    if load_note:
        return {"label": label, "engine": engine.name, "stand_in": is_stand_in, "failed": load_note, "events": [], "summary": None}
    for i, alert in enumerate(ALERTS):
        a = dict(alert)
        t0 = time.perf_counter()
        try:
            if lock:
                with lock:
                    res = engine.predict(render_state(a))
            else:
                res = engine.predict(render_state(a))
            ms = (time.perf_counter() - t0) * 1000
            ans = res["answers"]
            base = policy.decide(ans, a)
            action, _, _, corr = state.apply(base, a)
            out_tok += getattr(engine, "last_stats", {}).get("output_tokens", 0)
            ev = {"id": a["id"], "service": a["service"], "tier": a["tier"], "text": a["text"], "truth": a["truth"],
                  "action": action, "base_action": base["base_action"], "rule": base["rule"], "ms": round(ms, 1), "correct": policy.correct(action, a["truth"]),
                  "answers": {"category": ans["category"]["choice"],
                              **{k: round(ans[k]["noul"], 2) for k in ("customer_impact", "self_recovering", "sustained", "business_risk")},
                              "severity": round(ans["severity"]["score"], 2)},
                  "trace": explain.build_trace(ans, base, action, corr, a["truth"])}
            lat.append(ms)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            ev = {"id": a["id"], "service": a["service"], "tier": a["tier"], "text": a["text"], "truth": a["truth"],
                  "action": "ERROR", "rule": str(exc)[:160], "ms": round((time.perf_counter() - t0) * 1000, 1),
                  "correct": False, "answers": None}
        state.events.append(ev | {"decision": {"action": ev["action"]}})
        events.append(ev)
        progress(i + 1)
    scored = [e for e in events if e["truth"]]
    wrong = [e for e in scored if not e["correct"]]
    pages = [e for e in events if e["action"] == "PAGE"]
    missed = [e for e in scored if e["truth"] == "page" and e["action"] not in policy.PAGE_EQ]
    false_p = [e for e in scored if e["action"] == "PAGE" and e["truth"] != "page"]
    total_s = sum(lat) / 1000
    return {"label": label, "engine": engine.name, "stand_in": is_stand_in, "failed": None, "events": events,
            "summary": {"alerts": len(events), "errors": errors, "load_warmup_s": round(warm_s, 2),
                        "mean_ms": round(statistics.mean(lat), 1) if lat else None,
                        "p50_ms": round(_pct(lat, 50), 1), "p95_ms": round(_pct(lat, 95), 1), "max_ms": round(max(lat), 1) if lat else None,
                        "total_s": round(total_s, 2), "alerts_per_s": round(len(lat) / total_s, 2) if total_s else None,
                        "output_tokens": out_tok, "accuracy": round(1 - len(wrong) / len(scored), 3) if scored else None,
                        "correct": len(scored) - len(wrong), "scored": len(scored), "wrong": [e["id"] for e in wrong],
                        "missed_pages": [e["id"] for e in missed], "false_pages": [e["id"] for e in false_p],
                        "pages_sent": len(pages)}}


class CompareJob:
    """Background job: runs engines one after another, exposes progress + results for the UI."""

    def __init__(self):
        self.status = {"state": "idle", "done": 0, "total": 0, "phase": "", "error": None}
        self.result = None
        self._t = None

    def start(self, engines, lock=None):
        """engines: list of (label, engine, is_stand_in)."""
        if self.status["state"] == "running":
            return False
        self.status = {"state": "running", "done": 0, "total": len(ALERTS) * len(engines), "phase": "", "error": None,
                       "started": time.time()}
        self.result = None
        self._t = threading.Thread(target=self._run, args=(engines, lock), daemon=True)
        self._t.start()
        return True

    def _run(self, engines, lock):
        out = []
        try:
            for k, (label, eng, stand_in) in enumerate(engines):
                self.status["phase"] = "Running %s (%d/%d)" % (label, k + 1, len(engines))
                base = k * len(ALERTS)
                out.append(run_engine(label, eng, lambda d, b=base: self.status.update(done=b + d),
                                      lock=lock if eng.name == "laya" else None, is_stand_in=stand_in))
            self.result = {"engines": out, "finished": time.time(), "agreement": agreement(out)}
            self.status.update(state="done", done=self.status["total"], phase="Done")
        except Exception as exc:  # noqa: BLE001
            self.status.update(state="error", error="%s: %s" % (type(exc).__name__, exc))


def agreement(engines):
    """Share of alerts where engine[1] reached the same final decision as engine[0]."""
    if len(engines) < 2 or not engines[0]["events"] or not engines[1]["events"]:
        return None
    a, b = engines[0]["events"], engines[1]["events"]
    same = sum(1 for x, y in zip(a, b) if x["action"] == y["action"])
    return {"same": same, "total": len(a), "pct": round(100 * same / len(a), 1),
            "differ": [x["id"] for x, y in zip(a, b) if x["action"] != y["action"]]}
