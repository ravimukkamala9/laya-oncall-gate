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

    class FakeRouter:               # shape check for the real-Laya branch without weights
        device = "mps"
        loaded = ["english"]

    class FakeEngine:
        name, router = "laya", FakeRouter()
    monkeypatch.setattr(appmod, "ENGINE", FakeEngine())
    info = c.get("/api/engine").get_json()
    assert info["local"] is True and info["device"] == "mps" and info["checkpoints_loaded"] == ["english"]


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
    assert rules[0]["id"] == "R1" and rules[-1]["id"] == "R11" and len(rules) == 12


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
