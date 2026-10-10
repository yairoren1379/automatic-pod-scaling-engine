"""File-based persistence (no database): metrics time series as CSV, audit log as JSON lines."""
import csv
import json
import os
from collections import deque

METRICS_FIELDS = [
    "timestamp", "mode", "cpu_percent", "ram_percent", "latency_ms", "qps", "error_rate_percent",
    "replicas", "pod_crashed", "regime", "forecast_cpu_percent", "forecast_ram_percent",
    "forecast_latency_ms", "forecast_qps", "dqn_action", "executed_action", "source", "blocked",
    "reward",
]


class MetricsCsv:
    """One row per control loop. Used for the dashboard's history and to retrain the LSTM on
    real data (python train_lstm.py --from-csv data/metrics.csv)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=METRICS_FIELDS).writeheader()

    def append(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=METRICS_FIELDS, extrasaction="ignore").writerow(row)

    def tail(self, count: int) -> list:
        """The last `count` rows, converted back to numbers."""
        with open(self.path, newline="") as f:
            rows = deque(csv.DictReader(f), maxlen=count)
        return [_parse_row(row) for row in rows]


def _parse_row(row: dict) -> dict:
    parsed = {}
    for key, value in row.items():
        if key in ("mode", "dqn_action", "executed_action", "source"):
            parsed[key] = value
        elif key in ("pod_crashed", "blocked"):
            parsed[key] = value == "True"
        else:
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = None
    return parsed


class AuditLog:
    """The safety & audit log: every agent decision next to what the guardrails did with it."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, entry: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def tail(self, count: int) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            lines = deque(f, maxlen=count)
        return [json.loads(line) for line in lines if line.strip()]
