"""The classic threshold autoscaler ("scale up at 80% CPU"), the baseline the DQN must beat.

The Go controller implements the same rule (back/go/controller/baseline.go) for the real cluster.
"""
from core.actions import SCALE_UP, SCALE_DOWN, HOLD
from core.state import ClusterMetrics


def threshold_policy(metrics: ClusterMetrics, config: dict) -> int:
    baseline = config["baseline"]
    limits = config["guardrails"]

    if metrics.cpu_percent > baseline["scale_up_cpu_percent"] and metrics.replicas < limits["max_replicas"]:
        return SCALE_UP
    if metrics.cpu_percent < baseline["scale_down_cpu_percent"] and metrics.replicas > limits["min_replicas"]:
        return SCALE_DOWN
    return HOLD
