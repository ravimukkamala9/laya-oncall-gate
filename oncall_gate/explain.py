"""Turns one engine's answers + the policy walk into a step-by-step, human-readable trace.
Used by the Laya-vs-Ollama comparison so each row can show, per engine, HOW it reached its decision."""
from . import policy
from .questions import QUESTIONS

PAGE_RULES = ("R2", "R4", "R5", "R9")


def review_fields(a):
    """Laya's own uncertainty fields (laya >= 0.5); empty for engines that do not return them."""
    out = {k: a[k] for k in ("route", "route_reason", "abstain", "prediction_set", "unbound") if a.get(k) is not None}
    if a.get("level") is not None:
        out["level"] = a["level"]
    return out


def answer_rows(ans):
    rows = []
    for qid, q in QUESTIONS.items():
        a = ans[qid]
        if a["type"] == "choice":
            top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
            txt, frac = "%s (%s)" % (a["choice"], ", ".join("%s %.0f%%" % (k, v * 100) for k, v in top)), a["probabilities"][a["choice"]]
        elif a["type"] == "score":
            txt, frac = "%.2f of %d" % (a["score"], len(a["legend"]) - 1), a["score"] / (len(a["legend"]) - 1)
        else:
            txt, frac = "%.0f%% yes" % (a["noul"] * 100), a["noul"]
        rows.append({"id": qid, "question": q["instructions"], "answer": txt, "fraction": round(frac, 3), **review_fields(a)})
    return rows


def _gap(c):
    return "%s=%s (needs %s %s)" % (c["label"], policy._fmt(c["actual"]), c["op"], policy._fmt(c["thr"]))


def why(table, winner_action, final_action, truth, corr_steps):
    """Plain-language reasons: why the winner won, why earlier rules lost, and, if the result
    disagrees with the expected label, which PAGE rule came closest and what stopped it."""
    out = []
    win = next(r for r in table if r["winner"])
    held = " and ".join("%s=%s %s %s" % (c["label"], policy._fmt(c["actual"]), c["op"], policy._fmt(c["thr"]))
                        for c in win["conds"]) or "nothing earlier matched, so the default applied"
    out.append("%s \"%s\" matched first (%s) so the policy proposed %s." % (win["id"], win["name"], held, win["action"]))
    for r in table:
        if r is win:
            break
        bad = [c for c in r["conds"] if not c["ok"]]
        out.append("%s \"%s\" did not match: %s." % (r["id"], r["name"], "; ".join(_gap(c) for c in bad)))
    if final_action != winner_action:
        out.append("Correlation then changed %s into %s (see the correlation steps)." % (winner_action, final_action))
    if truth is not None and not policy.correct(final_action, truth):
        want = "PAGE or MERGED" if truth == "page" else truth.upper()
        out.append("Expected %s but got %s." % (want, final_action))
        if truth == "page":
            soft = ("customer_impact", "self_recovering", "sustained", "business_risk", "severity", "needs_review")
            cands = []
            for r in table:
                if r["id"] in PAGE_RULES:
                    bad = [c for c in r["conds"] if not c["ok"]]
                    if all(c["label"] in soft for c in bad):      # blocked only by a model answer, not by tier/category
                        cands.append((len(bad), sum(abs(c["actual"] - c["thr"]) / (3.0 if c["label"] == "severity" else 1.0) for c in bad), r, bad))
            if cands:
                _, _, r, bad = min(cands, key=lambda x: x[:2])
                out.append("Closest PAGE rule: %s \"%s\", blocked by %s." % (r["id"], r["name"], "; ".join(_gap(c) for c in bad)))
        else:
            if final_action == "PAGE":
                out.append("The rule that paged: %s \"%s\"." % (win["id"], win["name"]))
    return out


def build_trace(ans, base, action, corr, truth):
    return {"rows": answer_rows(ans), "table": base["table"], "headline": base["narrative"]["headline"],
            "base_action": base["base_action"], "corr_steps": corr["steps"],
            "why": why(base["table"], base["base_action"], action, truth, corr["steps"])}
