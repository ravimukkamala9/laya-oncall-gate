"""Scripted application alerts for the demo (t = minutes into the replay).

`truth` is the hidden ground-truth label used only for the scorecard:
page | watch | ticket | suppress.  Alerts that follow a page on the same service inside the
correlation window are expected to be MERGED into that incident (truth stays "page").
Tier 1 = customer-facing critical path, 2 = important internal, 3 = low criticality.
"""


def A(id, t, service, tier, source, text, truth, change=False):
    return {"id": id, "t": t, "service": service, "tier": tier, "source": source,
            "text": text, "truth": truth, "change_window": change}


ALERTS = [
    # --- noise / transient -----------------------------------------------------------------
    A("D1", 0, "orders-db", 1, "database", "Connection pool utilization 96% (baseline 40%), sustained for 4 minutes and rising. No customer impact reported yet.", "watch"),
    A("N1", 1, "notification-svc", 3, "infra", "CPU 92% for 20 seconds, auto-recovered. No customer impact.", "suppress"),
    A("M1", 2, "auth-service", 1, "memory", "JVM heap at 78%, rising slowly about 1% per 10 minutes. No customer impact.", "watch"),
    # --- database incident -----------------------------------------------------------------
    A("D2", 3, "orders-db", 1, "database", "Replication lag 92 seconds and worsening for 6 minutes; writes are delayed and order creation is failing for some customers.", "page"),
    A("N2", 4, "reporting-api", 3, "app", "Single request timeout, retry succeeded, self-resolved. No customer impact.", "suppress"),
    A("D3", 5, "orders-db", 1, "database", "Slow queries: p95 8 seconds, transaction commits timing out, customers affected.", "page"),
    A("N3", 6, "checkout-api", 1, "synthetic", "Health check blip: 1 of 3 probes failed, recovered on next probe. Transient, no customer impact.", "suppress"),
    A("D4", 7, "orders-db", 1, "database", "Failed transactions rising to 5% as commits time out; customers affected.", "page"),
    A("N4", 9, "batch-worker", 3, "memory", "Memory at 85% during nightly batch, transient and recovered within 30 seconds. No customer impact.", "suppress"),
    A("N5", 11, "reporting-api", 3, "app", "Single request timeout, retry succeeded, self-resolved. Duplicate of earlier alert.", "suppress"),
    # --- memory leak -----------------------------------------------------------------------
    A("M2", 14, "auth-service", 1, "memory", "Heap at 91% and still rising for 15 minutes; GC pauses of 2 seconds are slowing sign-ins and customers are seeing login delays.", "page"),
    A("M3", 16, "auth-service", 1, "memory", "Pod OOM-killed and restarting; login requests failing, customers affected.", "page"),
    # --- latency ---------------------------------------------------------------------------
    A("L1", 20, "checkout-api", 1, "latency", "p99 latency rose from 300 ms to 4.2 s over 8 minutes and is sustained; checkout pages are slow and customers are affected at peak load.", "page"),
    A("L2", 21, "reporting-api", 3, "latency", "p99 latency rose from 1.2 s to 3 s for 10 minutes and is rising. No customer impact; internal dashboards only.", "ticket"),
    # --- transaction failures / timeouts -----------------------------------------------------
    A("T1", 30, "order-service", 1, "transactions", "Transaction failure rate rose from 0.4% to 6% across multiple regions for 5 minutes and is sustained; customers affected.", "page"),
    A("T2", 32, "order-service", 1, "timeouts", "Downstream timeouts at 14% for one dependency, failed transactions rising, customers affected.", "page"),
    # --- approved maintenance ------------------------------------------------------------------
    A("C1", 40, "orders-db", 1, "database", "Primary failover in progress; replication lag spiking to 120 seconds. Approved change window CHG-4412 active.", "suppress", change=True),
    A("C2", 42, "orders-db", 1, "database", "Replica reconnecting, connections dropped briefly. Approved change window CHG-4412 active, expected.", "suppress", change=True),
    # --- IBM MQ -----------------------------------------------------------------------------
    A("Q1", 50, "mq-broker", 1, "mq", "Queue ORDERS.IN depth 12,000 and growing for 8 minutes, consumers 0, messages stuck; order processing delayed and customers affected.", "page"),
    A("Q2", 52, "mq-broker", 1, "mq", "Channel to downstream system retrying and stuck, queue manager failover not completing; customers affected.", "page"),
    A("Q3", 55, "report-etl", 2, "mq", "Dead-letter queue grew by 300 messages in 10 minutes and is rising. No customer impact.", "watch"),
    A("Q4", 58, "mq-broker", 1, "mq", "Channel retry flap, recovered within 15 seconds, transient. No customer impact.", "suppress"),
    # --- tickets ---------------------------------------------------------------------------
    A("K1", 60, "reporting-api", 3, "infra", "Disk usage 82% and growing slowly over 2 days. No customer impact.", "ticket"),
    A("K2", 62, "notification-svc", 3, "app", "Email queue backlog 5,000 messages and rising for 20 minutes; delivery to customers delayed.", "ticket"),
]
ALERTS.sort(key=lambda a: a["t"])
BY_ID = {a["id"]: a for a in ALERTS}
