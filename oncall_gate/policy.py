"""On-call paging policy. Laya supplies judgment (probabilities); thresholds and hard rules live
here, in code, because they must be auditable. A missed page costs more than an extra page, so
ties lean toward paging on tier-1 services."""
T = 0.5
WINDOW_MIN = 10          # correlation window (minutes)


_OPS = {">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "==": lambda a, b: a == b,
        "<=": lambda a, b: a <= b, "!=": lambda a, b: a != b}


def C(label, actual, op, thr):
    """One condition of a rule, with the actual value it was tested against."""
    return {"label": label, "actual": actual, "op": op, "thr": thr, "ok": bool(_OPS[op](actual, thr))}


MARGIN_T = 0.1           # fallback "unsure" line for engines without Laya's route field
# The yes/no answers that decide whether an alert pages. category and severity are left out on
# purpose: severity is already tested by value in R9, and uncalibrated Laya routes both of them to
# review on almost every alert (23 of the 24 scripted alerts), so their flag carries no information.
REVIEW_QS = ("customer_impact", "self_recovering", "sustained", "business_risk")


def category_margin(cat):
    """Top probability minus the runner-up. Computed from the probabilities because laya's own
    `answer_confidence` is the top probability (max p), not a margin."""
    ps = sorted((cat.get("probabilities") or {}).values(), reverse=True)
    if not ps:
        return 1.0
    return ps[0] - (ps[1] if len(ps) > 1 else 0.0)


def review_signal(a):
    """Is the engine itself unsure about this alert? Returns (needs_review, explanation).

    laya >= 0.5 answers carry `route` ("auto"/"review"), decided from the conformal prediction set
    and the abstain flag; Laya's guidance is to gate on that, not on a raw confidence number.
    Engines without it (classic Router, demo, Ollama) fall back to the category margin."""
    routed = [q for q in REVIEW_QS if a.get(q, {}).get("route") is not None]
    if routed:
        flagged = [q for q in routed if a[q]["route"] == "review"]
        return bool(flagged), ("Laya route=review on %s" % ", ".join(flagged) if flagged
                               else "Laya route=auto on %s" % ", ".join(routed))
    margin = category_margin(a["category"])
    return margin < MARGIN_T, "category margin %.2f %s %.2f (no Laya route field)" % (
        margin, "<" if margin < MARGIN_T else ">=", MARGIN_T)


def _rules(a, alert):
    """THE decision table: ordered rules, first match wins. Used for the decision AND for the UI."""
    cat = a["category"]["choice"]
    imp, rec = a["customer_impact"]["noul"], a["self_recovering"]["noul"]
    sus, risk = a["sustained"]["noul"], a["business_risk"]["noul"]
    sev, tier = a["severity"]["score"], alert["tier"]
    unsure, unsure_why = review_signal(a)
    return [
        dict(id="R1", name="maintenance window", action="SUPPRESS",
             conds=[C("change_window", bool(alert.get("change_window")), "==", True)],
             why="approved change window active"),
        dict(id="R2", name="tier-1 transaction failures", action="PAGE",
             conds=[C("tier", tier, "==", 1), C("category", cat, "==", "transaction_failure"),
                    C("customer_impact", imp, ">=", T), C("self_recovering", rec, "<", T)],
             why="tier-1 transaction failure, impact=%.2f" % imp),
        dict(id="R3", name="self-recovering", action="SUPPRESS",
             conds=[C("self_recovering", rec, ">=", T), C("sustained", sus, "<", T)],
             why="self_recovering=%.2f, sustained=%.2f" % (rec, sus)),
        dict(id="R4", name="customer impact + severity", action="PAGE",
             conds=[C("customer_impact", imp, ">=", T), C("severity", sev, ">=", 2), C("tier", tier, "<=", 2)],
             why="impact=%.2f severity=%.2f tier-%d" % (imp, sev, tier)),
        dict(id="R4b", name="tier-3 impact", action="TICKET",
             conds=[C("customer_impact", imp, ">=", T), C("severity", sev, ">=", 2), C("tier", tier, "==", 3)],
             why="tier-3 service, impact=%.2f" % imp),
        dict(id="R5", name="business risk on tier-1", action="PAGE",
             conds=[C("business_risk", risk, ">=", T), C("tier", tier, "==", 1), C("sustained", sus, ">=", 0.4)],
             why="business_risk=%.2f sustained=%.2f" % (risk, sus)),
        dict(id="R6", name="impact, lower severity", action="TICKET",
             conds=[C("customer_impact", imp, ">=", T)],
             why="impact=%.2f severity=%.2f" % (imp, sev)),
        dict(id="R7", name="tier-3 sustained", action="TICKET",
             conds=[C("sustained", sus, ">=", T), C("tier", tier, "==", 3)],
             why="sustained=%.2f, tier-3, no impact" % sus),
        dict(id="R8", name="sustained, no impact yet", action="WATCH",
             conds=[C("sustained", sus, ">=", T), C("tier", tier, "<", 3)],
             why="sustained=%.2f impact=%.2f" % (sus, imp)),
        dict(id="R9", name="model unsure on tier-1: lean to page", action="PAGE",
             conds=[C("tier", tier, "==", 1), C("severity", sev, ">=", 1.5), C("needs_review", unsure, "==", True)],
             why="severity=%.2f, %s" % (sev, unsure_why)),
        dict(id="R9b", name="tier-1 safety net", action="PAGE",
             conds=[C("tier", tier, "==", 1), C("severity", sev, ">=", 1.5)],
             why="tier-1, severity=%.2f, no earlier rule matched: a missed page costs more than an extra one" % sev),
        dict(id="R10", name="default", action="TICKET",
             conds=[C("severity", sev, ">=", 1)], why="severity=%.2f" % sev),
        dict(id="R11", name="negligible", action="SUPPRESS", conds=[], why="severity=%.2f" % sev),
    ]


def _fmt(v):
    return "%.2f" % v if isinstance(v, float) else str(v)


def decide(a, alert):
    """Walk the rule table top to bottom; the first rule whose conditions ALL hold wins."""
    table, winner = _rules(a, alert), None
    for r in table:
        r["matched"] = all(c["ok"] for c in r["conds"])
        r["reached"] = winner is None
        if r["matched"] and winner is None:
            winner = r
        r["winner"] = r is winner
    return {"base_action": winner["action"], "rule": "%s %s" % (winner["id"], winner["name"]),
            "reasons": [winner["why"]], "table": table, "narrative": narrative(table, winner)}


def narrative(table, winner):
    won = " and ".join("%s=%s %s %s" % (c["label"], _fmt(c["actual"]), c["op"], _fmt(c["thr"]))
                       for c in winner["conds"]) or "no earlier rule matched (default)"
    skipped = []
    for r in table:
        if r is winner:
            break
        bad = next(c for c in r["conds"] if not c["ok"])
        skipped.append("%s (%s=%s, needs %s %s)" % (r["id"], bad["label"], _fmt(bad["actual"]), bad["op"], _fmt(bad["thr"])))
    return {"headline": "%s wins: %s -> %s" % (winner["id"], won, winner["action"]),
            "earlier_rules_rejected": skipped}


def rules_doc():
    """Static view of the decision table for the Architecture tab (derived from the same code)."""
    dummy = {"category": {"choice": "x", "answer_confidence": 1.0}, "customer_impact": {"noul": 0.0},
             "self_recovering": {"noul": 0.0}, "sustained": {"noul": 0.0}, "business_risk": {"noul": 0.0},
             "severity": {"score": 0.0}}
    return [{"id": r["id"], "name": r["name"], "action": r["action"],
             "conditions": ["%s %s %s" % (c["label"], c["op"], _fmt(c["thr"])) for c in r["conds"]] or ["always (default)"]}
            for r in _rules(dummy, {"tier": 1})]


class OnCallState:
    """Correlation + simulated pager. One incident per service inside WINDOW_MIN minutes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.events, self.pages, self.incidents, self.watch = [], [], {}, {}
        self.clock = 0

    def apply(self, base, alert):
        t, svc = alert["t"], alert["service"]
        self.clock = max(self.clock, t)
        inc = self.incidents.get(svc)
        open_inc = bool(inc and t - inc["last"] <= WINDOW_MIN)
        action, note, steps = base["base_action"], None, []
        steps.append("Policy proposed %s." % action)
        if inc:
            steps.append("Incident %s exists for %s, last activity %d min ago -> %s (window %d min)." % (
                inc["id"], svc, t - inc["last"], "still OPEN" if open_inc else "closed", WINDOW_MIN))
        else:
            steps.append("No earlier incident for %s." % svc)
        prior_watch = None
        if action == "WATCH":
            w = self.watch.get(svc)
            if w is None:
                steps.append("First WATCH signal for %s: remembered at t+%dm." % (svc, t))
            else:
                prior_watch = t - w
                if prior_watch <= WINDOW_MIN and not open_inc:
                    action, note = "PAGE", "watch escalated: second sustained signal on same service"
                    steps.append("Earlier WATCH %d min ago (<= %d) -> escalate to PAGE." % (prior_watch, WINDOW_MIN))
                else:
                    steps.append("Earlier WATCH %d min ago is outside the window or an incident is open -> stays WATCH." % prior_watch)
            self.watch[svc] = t
        if action == "PAGE" and open_inc:
            action, note = "MERGED", "joined %s (no new page)" % inc["id"]
            steps.append("A page already covers this service -> MERGED into %s (no second page)." % inc["id"])
        elif action == "TICKET" and open_inc:
            action, note = "MERGED", "related to open incident %s" % inc["id"]
            steps.append("Open incident on same service -> ticket merged into %s." % inc["id"])
        if action == "PAGE":
            inc = {"id": "INC-%03d" % (len(self.incidents) + 1), "service": svc, "opened": t,
                   "last": t, "alerts": [alert["id"]]}
            self.incidents[svc] = inc
            self.pages.append({"t": t, "incident": inc["id"], "service": svc,
                               "message": "PAGE on-call (%s): %s" % (svc, alert["text"][:110])})
            steps.append("New incident %s opened and a page written to the simulated pager." % inc["id"])
        elif action == "MERGED":
            inc["last"] = t
            inc["alerts"].append(alert["id"])
        corr = {"window_min": WINDOW_MIN, "open_incident": inc["id"] if (open_inc or action == "PAGE") and inc else None,
                "prior_watch_min_ago": prior_watch, "steps": steps}
        return action, note, (inc["id"] if inc and action in ("PAGE", "MERGED") else None), corr


PAGE_EQ = {"PAGE", "MERGED"}


def correct(action, truth):
    if truth is None:
        return None
    return (action in PAGE_EQ) if truth == "page" else action.lower() == truth


def scorecard(events):
    scored = [e for e in events if e["truth"]]
    pages = [e for e in events if e["decision"]["action"] == "PAGE"]
    missed = [e for e in scored if e["truth"] == "page" and e["decision"]["action"] not in PAGE_EQ]
    false_p = [e for e in scored if e["decision"]["action"] == "PAGE" and e["truth"] != "page"]
    wrong = [e for e in scored if not e["correct"]]
    n = len(events)
    return {"alerts": n, "pages_sent": len(pages), "pages_avoided": max(0, n - len(pages)),
            "by_decision": {k: sum(1 for e in events if e["decision"]["action"] == k)
                            for k in ("PAGE", "MERGED", "WATCH", "TICKET", "SUPPRESS")},
            "scored": len(scored), "accuracy": (1 - len(wrong) / len(scored)) if scored else None,
            "missed_pages": [e["id"] for e in missed], "false_pages": [e["id"] for e in false_p],
            "wrong": [e["id"] for e in wrong]}
