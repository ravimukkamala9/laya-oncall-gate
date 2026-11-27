"""The decision schema: what we ask Laya about every alert, in ONE forward pass.

Three Laya primitives: choice (pick a label), score (ordinal level), noul (yes/no probability).
"""

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What kind of problem does `alert` describe?",
        "criteria": {
            "database": "database connections, replication lag, slow queries, locks",
            "transaction_failure": "failed transactions, error-rate spikes, failed business operations",
            "memory": "heap or memory growth, GC pauses, out-of-memory kills",
            "latency": "response-time slowness, high p95 or p99 latency",
            "timeout": "request or downstream timeouts",
            "messaging": "message queue depth, channel or queue manager problems, dead-letter queue growth",
            "infra_other": "CPU, disk, health checks or anything else",
        },
    },
    "customer_impact": {"type": "noul", "instructions": "Are customers or users being affected right now?"},
    "self_recovering": {"type": "noul",
                        "instructions": "Is this transient, already recovered, a known flap or expected maintenance?"},
    "sustained": {"type": "noul", "instructions": "Is the problem sustained or getting worse over several minutes?"},
    "business_risk": {"type": "noul",
                      "instructions": "Is there risk of lost or failed transactions or data integrity problems?"},
    "severity": {"type": "score", "instructions": "How severe is the operational impact?",
                 "criteria": ["negligible", "minor", "significant", "critical"]},
}
