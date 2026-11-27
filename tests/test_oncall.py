import os
os.environ["LAYA_ENGINE"] = "demo"
from app import app
from oncall_gate.alerts import ALERTS


def replay():
    c = app.test_client()
    c.post("/api/reset")
    ev = {a["id"]: c.post("/api/alert", json={"id": a["id"]}).get_json() for a in ALERTS}
    return c, ev


def test_noise_never_pages():
    _, ev = replay()
    for i in ("N1", "N2", "N3", "N4", "N5", "Q4"):
        assert ev[i]["decision"]["action"] == "SUPPRESS", i


def test_maintenance_suppressed():
    _, ev = replay()
    assert ev["C1"]["decision"]["action"] == ev["C2"]["decision"]["action"] == "SUPPRESS"


def test_db_incident_pages_once():
    _, ev = replay()
    assert ev["D2"]["decision"]["action"] == "PAGE"
    assert ev["D3"]["decision"]["action"] == ev["D4"]["decision"]["action"] == "MERGED"


def test_no_missed_pages_and_fewer_pages_than_alerts():
    c, _ = replay()
    s = c.get("/api/scorecard").get_json()
    assert s["missed_pages"] == [] and s["false_pages"] == []
    assert s["pages_sent"] < s["alerts"] / 3
    assert s["accuracy"] >= 0.9


def test_custom_alert():
    c = app.test_client()
    c.post("/api/reset")
    e = c.post("/api/alert", json={"service": "x", "tier": 1,
               "text": "Connection pool at 99% for 6 minutes and rising; order creation failing for customers."}).get_json()
    assert e["decision"]["action"] == "PAGE"
    assert c.post("/api/alert", json={}).status_code == 400
    assert c.get("/").status_code == 200 and c.get("/api/health").get_json()["engine"] == "demo"


def test_engine_info_demo_and_laya_shape(monkeypatch):
    import app as appmod
    c = app.test_client()
    info = c.get("/api/engine").get_json()
    assert info["engine"] == "demo" and info["local"] is False

    class FakeEngine:               # shape check for the real-Laya branch without weights
        name = "laya"

        def describe(self):
            return {"backend": "vega", "model": "vega", "model_label": "laya.Agent()",
                    "device": "mps", "checkpoints_loaded": ["nandakishorm/vega-U"]}
    monkeypatch.setattr(appmod, "ENGINE", FakeEngine())
    info = c.get("/api/engine").get_json()
    assert info["local"] is True and info["device"] == "mps" and info["checkpoints_loaded"] == ["nandakishorm/vega-U"]
    assert info["backend"] == "vega"

def test_event_has_trace_fields():
    c = app.test_client()
    c.post("/api/reset")
    e = c.post("/api/alert", json={"id": "D2"}).get_json()
    assert e["state_sent"]["alert"] and "category_p" in e["answers"]


def test_decision_path_and_table():
    c = app.test_client()
    c.post("/api/reset")
    e = c.post("/api/alert", json={"service": "Oders-db", "tier": 1,
               "text": "Connection pool is at 99% usage and its rising for last 6 mins."}).get_json()
    steps = e["path"]["steps"]
    assert [s["n"] for s in steps] == [1, 2, 3, 4, 5, 6]
    table = steps[3]["table"]
    winners = [r for r in table if r["winner"]]
    assert len(winners) == 1                                   # exactly one winning rule
    assert winners[0]["action"] == e["decision"]["base_action"]
    first_match = next(r for r in table if r["matched"])       # first match wins
    assert first_match is winners[0]
    assert all(not r["reached"] for r in table[table.index(winners[0]) + 1:])
    assert len(steps[2]["rows"]) == 6 and steps[3]["loc"].startswith("oncall_gate/policy.py")


def test_policy_doc():
    rules = app.test_client().get("/api/policy").get_json()
    assert rules[0]["id"] == "R1" and rules[-1]["id"] == "R11" and len(rules) == 13


# ---------------------------------------------------------------- Ollama comparison (mock server)
def _mock(port, **kw):
    import mock_ollama
    for k, v in kw.items():
        setattr(mock_ollama.H, k, v)
    return mock_ollama.start(port)


def test_ollama_engine_shape_and_policy_runs():
    from oncall_gate.ollama_engine import OllamaEngine, list_models
    from oncall_gate import policy
    from oncall_gate.engine import render_state
    from oncall_gate.alerts import BY_ID
    srv = _mock(11991, delay=0.01, broken_every=0)
    try:
        assert "llama3.2:latest" in list_models("http://127.0.0.1:11991")
        eng = OllamaEngine("llama3.2", "http://127.0.0.1:11991")
        res = eng.predict(render_state(BY_ID["D2"]))
        assert set(res["answers"]) == {"category", "customer_impact", "self_recovering", "sustained", "business_risk", "severity"}
        assert eng.last_stats["output_tokens"] == 58
        assert policy.decide(res["answers"], BY_ID["D2"])["base_action"] in ("PAGE", "WATCH", "TICKET", "SUPPRESS")
    finally:
        srv.shutdown()


def test_ollama_unreachable_gives_clear_error():
    import pytest
    from oncall_gate.ollama_engine import OllamaEngine, OllamaError
    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        OllamaEngine("x", "http://127.0.0.1:9").predict({"service": "s", "tier": "t", "source": "x", "alert": "a"})


def test_compare_api_end_to_end_with_retry():
    import time as t
    srv = _mock(11992, delay=0.01, broken_every=7)
    c = app.test_client()
    try:
        assert c.get("/api/ollama/models?url=http://127.0.0.1:11992").get_json()["ok"] is True
        assert c.get("/api/ollama/models?url=http://127.0.0.1:9").get_json()["ok"] is False
        assert c.post("/api/compare/start", json={"url": "http://127.0.0.1:11992", "model": "llama3.2"}).get_json()["ok"]
        for _ in range(200):
            st = c.get("/api/compare/status").get_json()
            if st["status"]["state"] != "running":
                break
            t.sleep(0.1)
        assert st["status"]["state"] == "done", st["status"]
        eng = st["result"]["engines"]
        assert len(eng) == 2 and eng[0]["stand_in"] is True and eng[1]["engine"] == "ollama"
        assert eng[1]["summary"]["alerts"] == 24 and eng[1]["summary"]["mean_ms"] > 0
        import mock_ollama
        assert eng[1]["summary"]["errors"] == 0 and mock_ollama.H.calls > 25   # one-off bad JSON was retried and recovered
        assert st["result"]["agreement"]["total"] == 24
    finally:
        srv.shutdown()


def test_persistently_malformed_output_is_reported_not_hidden():
    from oncall_gate.compare import run_engine
    from oncall_gate.ollama_engine import OllamaEngine
    srv = _mock(11993, delay=0.0, broken_every=1)                  # every reply is garbage
    try:
        r = run_engine("Ollama bad", OllamaEngine("m", "http://127.0.0.1:11993"), lambda d: None)
        assert r["failed"] and "malformed" in r["failed"]          # surfaced as a failure with the reason
    finally:
        srv.shutdown()


def test_compare_events_carry_step_by_step_trace():
    from oncall_gate.compare import run_engine
    from oncall_gate.engine import DemoEngine
    r = run_engine("Demo", DemoEngine(), lambda d: None, is_stand_in=True)
    for e in r["events"]:
        t = e["trace"]
        assert len(t["rows"]) == 6 and t["table"] and t["corr_steps"] and t["why"]
        assert sum(1 for x in t["table"] if x["winner"]) == 1
        assert e["base_action"] == next(x for x in t["table"] if x["winner"])["action"]
    wrong = [e for e in r["events"] if e["correct"] is False]
    for e in wrong:
        assert any(w.startswith("Expected") for w in e["trace"]["why"])


def test_why_names_closest_page_rule_for_missed_page():
    from oncall_gate import explain, policy
    ans = {"category": {"type": "choice", "choice": "database", "probabilities": {"database": 1.0}, "confidence": 1.0, "answer_confidence": 1.0},
           "customer_impact": {"type": "noul", "noul": 0.7}, "self_recovering": {"type": "noul", "noul": 0.1},
           "sustained": {"type": "noul", "noul": 0.8}, "business_risk": {"type": "noul", "noul": 0.2},
           "severity": {"type": "score", "score": 1.6, "legend": ["a", "b", "c", "d"]}}
    alert = {"tier": 1, "id": "D2", "service": "orders-db", "t": 1, "text": "x", "truth": "page"}
    base = policy.decide(ans, alert)
    assert base["base_action"] == "TICKET"
    why = explain.why(base["table"], "TICKET", "TICKET", "page", [])
    assert any("Closest PAGE rule: R4" in w and "severity=1.60" in w for w in why)
    assert any("R9b" in w and "R6" in w and "won" in w for w in why)     # matched but pre-empted, not "blocked"


def test_safety_net_pages_tier1_when_nothing_else_matches():
    from oncall_gate import policy
    a = _vega_answers(route="auto", sev_expected=1.7)              # model sure, no impact, no rule above matches
    d = policy.decide(a, {"tier": 1})
    assert d["rule"].split()[0] == "R9b" and d["base_action"] == "PAGE"
    assert policy.decide(a, {"tier": 2})["base_action"] != "PAGE"   # tier-1 only
    a = _vega_answers(route="auto", sev_expected=1.2)
    assert policy.decide(a, {"tier": 1})["rule"].split()[0] != "R9b"


# ---------------------------------------------------------------- laya >= 0.5 answer shape (no weights needed)
def _vega_answers(route="auto", sev_level=2, sev_expected=1.7, margin=0.9):
    """Answers shaped like laya.Agent() (vega engine) returns them, after engine.normalise()."""
    def noul(p):
        return {"type": "noul", "noul": p, "probabilities": {"false": 1 - p, "true": p},
                "prediction_set": ["true" if p >= 0.5 else "false"], "abstain": False, "unbound": False,
                "route": route, "route_reason": "test"}
    res = {"answers": {
        "category": {"type": "choice", "choice": "database",
                     "probabilities": {"database": (1 + margin) / 2, "memory": (1 - margin) / 2},
                     "answer_confidence": (1 + margin) / 2, "prediction_set": ["database"], "abstain": False, "route": route},
        "customer_impact": noul(0.3), "self_recovering": noul(0.1), "sustained": noul(0.3), "business_risk": noul(0.2),
        "severity": {"type": "score", "score": sev_level, "expected_score": sev_expected,
                     "legend": {"0": "negligible", "1": "minor", "2": "significant", "3": "critical"},
                     "prediction_set": ["1", "2"], "abstain": False, "route": route}}}
    from oncall_gate.engine import normalise
    return normalise(res)["answers"]


def test_normalise_uses_expected_score_for_severity():
    a = _vega_answers(sev_level=2, sev_expected=1.7)
    assert a["severity"]["score"] == 1.7 and a["severity"]["level"] == 2


def test_r9_follows_laya_route_not_margin():
    from oncall_gate import policy
    alert = {"tier": 1}
    # Laya says review -> R9 pages even though the category margin is high
    d = policy.decide(_vega_answers(route="review", margin=0.9), alert)
    assert d["rule"].split()[0] == "R9" and d["base_action"] == "PAGE"
    assert "route=review" in d["reasons"][0]
    # Laya says auto -> R9 does not fire even though the margin is low
    d = policy.decide(_vega_answers(route="auto", margin=0.01), alert)
    assert d["rule"].split()[0] != "R9"


def test_r9_falls_back_to_margin_without_route():
    from oncall_gate import policy
    a = _vega_answers(margin=0.05)
    for q in a.values():
        q.pop("route", None)
    unsure, why = policy.review_signal(a)
    assert unsure is True and "margin" in why
    assert policy.decide(a, {"tier": 1})["rule"].split()[0] == "R9"


def test_trace_rows_carry_laya_review_fields():
    from oncall_gate import explain
    rows = explain.answer_rows(_vega_answers(route="review"))
    sev = next(r for r in rows if r["id"] == "severity")
    assert sev["route"] == "review" and sev["prediction_set"] == ["1", "2"] and sev["level"] == 2


def test_laya_engine_rejects_unknown_model():
    import pytest
    from oncall_gate.engine import LayaEngine
    with pytest.raises(ValueError, match="LAYA_MODEL"):
        LayaEngine("nope")


def test_r9_ignores_review_on_category_and_severity_only():
    """Uncalibrated Laya flags category and severity on almost every alert; that alone must not page."""
    from oncall_gate import policy
    a = _vega_answers(route="auto")
    a["category"]["route"] = a["severity"]["route"] = "review"
    assert policy.review_signal(a)[0] is False
    assert policy.decide(a, {"tier": 1})["rule"].split()[0] != "R9"


def test_margin_fallback_uses_probabilities_not_answer_confidence():
    """laya's classic answer_confidence is max(p) (always >= 1/7 here); the margin must be top - second."""
    from oncall_gate import policy
    a = _vega_answers()
    for q in a.values():
        q.pop("route", None)
    a["category"]["probabilities"] = {"latency": 0.30, "database": 0.28, "memory": 0.22, "timeout": 0.20}
    a["category"]["answer_confidence"] = 0.30
    unsure, why = policy.review_signal(a)
    assert unsure is True and "0.02" in why
    assert policy.decide(a, {"tier": 1})["rule"].split()[0] == "R9"


def test_flagged_matches_the_questions_r9_uses(monkeypatch):
    import app as appmod
    from oncall_gate.engine import DemoEngine

    class Vegaish(DemoEngine):
        name = "laya"

        def predict(self, state):
            res = DemoEngine.predict(self, state)
            for q in res["answers"].values():
                q["route"] = "auto"
            res["answers"]["category"]["route"] = "review"      # not a question R9 uses
            return res
    monkeypatch.setattr(appmod, "ENGINE", Vegaish())
    c = appmod.app.test_client()
    c.post("/api/reset")
    e = c.post("/api/alert", json={"id": "D2"}).get_json()
    assert e["review"] == {"needs_review": False, "flagged": [], "why": e["review"]["why"]}
