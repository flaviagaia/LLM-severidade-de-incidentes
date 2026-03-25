from __future__ import annotations

import json
import random
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASET_PATH, LABELS, PROCESSED_DIR, RAW_DIR, SUMMARY_PATH, TEST_PATH, TRAIN_PATH, VALID_PATH


@dataclass(frozen=True)
class IncidentPattern:
    system: str
    location: str
    symptom: str
    impact: str
    signal: str
    severity: str


CRITICAL_PATTERNS = [
    IncidentPattern("payment gateway", "primary datacenter", "complete outage", "transactions are blocked for all customers", "executive bridge activated", "critical"),
    IncidentPattern("core banking API", "production cluster", "timeouts above 90 percent", "branch operations cannot authenticate", "regulatory SLA breached", "critical"),
    IncidentPattern("risk engine", "main region", "pipeline stopped", "fraud screening is unavailable", "critical financial exposure", "critical"),
    IncidentPattern("ATM network", "national backbone", "communication lost", "cash withdrawal service unavailable nationwide", "all channels impacted", "critical"),
]

HIGH_PATTERNS = [
    IncidentPattern("loan approval service", "regional cluster", "latency spike", "requests are delayed for several teams", "multiple escalations opened", "high"),
    IncidentPattern("customer portal", "internet-facing zone", "intermittent failures", "users report repeated retries", "support backlog increasing", "high"),
    IncidentPattern("document archive", "storage tier", "replication lag", "analysts cannot access recent files", "manual workaround required", "high"),
    IncidentPattern("card issuance batch", "night processing queue", "job retries", "issuance volume is below expectation", "overnight deadline at risk", "high"),
]

NORMAL_PATTERNS = [
    IncidentPattern("internal reporting dashboard", "analytics workspace", "slow refresh", "a small user group is affected", "temporary workaround available", "normal"),
    IncidentPattern("notification service", "secondary region", "minor delay", "messages are delivered with low impact", "no external SLA breach", "normal"),
    IncidentPattern("document signing portal", "office subnet", "single user error", "one workstation cannot complete the task", "service remains available", "normal"),
    IncidentPattern("inventory sync", "backoffice integration", "warning alerts", "data is delayed but consistent", "business process continues", "normal"),
]

TEAMS = ["payments", "operations", "security", "network", "digital channels", "compliance"]
SHIFTS = ["morning shift", "afternoon shift", "night shift", "weekend shift"]
TIMINGS = ["for 5 minutes", "for 20 minutes", "for 45 minutes", "for over an hour"]


def _compose_text(pattern: IncidentPattern, rng: random.Random) -> str:
    team = rng.choice(TEAMS)
    shift = rng.choice(SHIFTS)
    timing = rng.choice(TIMINGS)
    return (
        f"Incident report: the {pattern.system} in the {pattern.location} shows {pattern.symptom}. "
        f"Impact: {pattern.impact}. Additional note: {pattern.signal}. "
        f"Reported by the {team} team during the {shift} and observed {timing}."
    )


def build_synthetic_dataset(rows_per_label: int = 320, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    patterns = {
        "critical": CRITICAL_PATTERNS,
        "high": HIGH_PATTERNS,
        "normal": NORMAL_PATTERNS,
    }
    rows: list[dict] = []
    for severity in LABELS:
        available = patterns[severity]
        for idx in range(rows_per_label):
            pattern = rng.choice(available)
            rows.append(
                {
                    "incident_id": f"INC-{severity[:1].upper()}{idx:04d}",
                    "incident_text": _compose_text(pattern, rng),
                    "severity": severity,
                    "system": pattern.system,
                    "location": pattern.location,
                }
            )
    return pd.DataFrame(rows)


def prepare_dataset(rows_per_label: int = 320) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_synthetic_dataset(rows_per_label=rows_per_label)
    dataset.to_csv(DATASET_PATH, index=False)

    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.30,
        random_state=42,
        stratify=dataset["severity"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["severity"],
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    valid_df.to_csv(VALID_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    summary = {
        "rows": int(len(dataset)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "labels": LABELS,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
