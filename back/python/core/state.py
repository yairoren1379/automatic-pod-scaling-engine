"""Cluster metrics and how they become the neural networks' input vectors."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterMetrics:
    """One snapshot of the target Deployment, taken once per control loop."""
    cpu_percent: float          # average CPU of the ready pods, 0-100 (of the pod limit)
    ram_percent: float          # average memory of the ready pods, 0-100 (of the pod limit)
    latency_ms: float           # average request latency
    qps: float                  # total requests per second reaching the service
    error_rate_percent: float   # share of failed requests, 0-100
    replicas: int               # desired replica count of the Deployment
    pod_crashed: bool = False   # a pod crashed or went NotReady during the last step


# The metrics the LSTM forecasts, in this order.
FORECAST_METRICS = ["cpu_percent", "ram_percent", "latency_ms", "qps"]

# Size of normalize_metrics(): cpu, ram, latency, qps, error rate, replicas.
METRICS_VECTOR_SIZE = 6

# The DQN's input: the 6 (smoothed) metrics + 4 LSTM forecasts + the hysteresis load regime.
STATE_SIZE = METRICS_VECTOR_SIZE + len(FORECAST_METRICS) + 1


def _scales(config: dict) -> dict:
    """The value each metric is divided by. Inputs above the scale are clipped (see below)."""
    norm = config["state_normalization"]
    return {
        "cpu_percent": 100.0,
        "ram_percent": 100.0,
        "latency_ms": norm["latency_scale_ms"],
        "qps": norm["qps_scale"],
        "error_rate_percent": norm["error_scale_percent"],
        "replicas": config["guardrails"]["max_replicas"],
    }


# Upper clip per metric (in normalized units), so one extreme reading can't dominate the network.
_CLIP = {"latency_ms": 3.0, "qps": 3.0, "error_rate_percent": 5.0}


def normalize_metrics(metrics: ClusterMetrics, config: dict) -> np.ndarray:
    """Scales every metric to roughly [0, 1].

    Neural networks learn badly when inputs have very different ranges (QPS in the hundreds,
    error rate near zero), so each metric is divided by a typical maximum.
    """
    scales = _scales(config)
    names = FORECAST_METRICS + ["error_rate_percent", "replicas"]
    values = []
    for name in names:
        value = getattr(metrics, name) / scales[name]
        values.append(min(value, _CLIP.get(name, value)))
    return np.array(values, dtype=np.float32)


def normalize_forecast(forecast: dict, config: dict) -> np.ndarray:
    scales = _scales(config)
    return np.array([min(forecast[name] / scales[name], _CLIP.get(name, 10.0))
                     for name in FORECAST_METRICS], dtype=np.float32)


def denormalize_forecast(vector, config: dict) -> dict:
    """The inverse of normalize_forecast: from network units back to percent / ms / QPS."""
    scales = _scales(config)
    return {name: max(0.0, float(value) * scales[name]) for name, value in zip(FORECAST_METRICS, vector)}


def build_state_vector(smoothed: ClusterMetrics, forecast: dict, regime: int, config: dict) -> np.ndarray:
    """The DQN input: where the cluster is now, where it is heading, and its load regime."""
    return np.concatenate([
        normalize_metrics(smoothed, config),
        normalize_forecast(forecast, config),
        np.array([regime], dtype=np.float32),   # -1 low load, 0 normal, +1 high load
    ])
