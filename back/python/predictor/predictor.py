"""The Predictor: turns raw metrics into the DQN state.

    raw metrics -> moving average -> smoothed metrics -----------------------> state
                                         |-> hysteresis on CPU -> load regime -> state
                                         |-> history window -> LSTM -> forecasts -> state

The same class runs in training (mock environment) and in the API server (real cluster), so
the agent always sees states built the same way.
"""
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import torch

from core.state import (ClusterMetrics, FORECAST_METRICS, build_state_vector,
                        denormalize_forecast, normalize_metrics)
from predictor.lstm_forecaster import LoadForecaster, load_forecaster
from predictor.smoothing import HysteresisRegime, MovingAverage


@dataclass
class PredictorOutput:
    smoothed: ClusterMetrics    # the noise-filtered current metrics
    forecast: dict              # cpu_percent, ram_percent, latency_ms, qps `horizon` steps ahead
    regime: int                 # -1 low load, 0 normal, +1 high load
    state: np.ndarray           # the DQN input built from all of the above


class Predictor:

    def __init__(self, config: dict, forecaster: LoadForecaster = None, forecaster_meta: dict = None):
        """Without a forecaster, the forecast is the naive one: the future equals now."""
        self.config = config
        predictor_cfg = config["predictor"]
        self.smoother = MovingAverage(predictor_cfg["smoothing_window"])
        self.hysteresis = HysteresisRegime(
            high_enter=predictor_cfg["high_enter_cpu_percent"],
            high_exit=predictor_cfg["high_exit_cpu_percent"],
            low_enter=predictor_cfg["low_enter_cpu_percent"],
            low_exit=predictor_cfg["low_exit_cpu_percent"],
        )
        self.forecaster = forecaster
        self.sequence_length = (forecaster_meta or {}).get(
            "sequence_length", predictor_cfg["sequence_length"])
        self.history = deque(maxlen=self.sequence_length)

    @classmethod
    def from_checkpoint(cls, config: dict, lstm_path: str = None):
        if lstm_path is None:
            return cls(config)
        model, meta = load_forecaster(lstm_path)
        return cls(config, model, meta)

    def reset(self):
        """Forget the history (a new episode, or a gap in the real metrics)."""
        self.smoother.reset()
        self.hysteresis.reset()
        self.history.clear()

    def observe(self, metrics: ClusterMetrics) -> PredictorOutput:
        # Smooth the noisy gauges. Error rate, replicas and crashes are kept as they are:
        # they are exact, and a crash must not be averaged away.
        smoothed_values = self.smoother.update(
            [metrics.cpu_percent, metrics.ram_percent, metrics.latency_ms, metrics.qps])
        smoothed = replace(metrics, **dict(zip(FORECAST_METRICS, map(float, smoothed_values))))

        regime = self.hysteresis.update(smoothed.cpu_percent)

        vector = normalize_metrics(smoothed, self.config)
        self.history.append(vector)
        forecast = self._forecast(vector)

        state = build_state_vector(smoothed, forecast, regime, self.config)
        return PredictorOutput(smoothed=smoothed, forecast=forecast, regime=regime, state=state)

    def _forecast(self, current_vector: np.ndarray) -> dict:
        current = current_vector[:len(FORECAST_METRICS)]
        if self.forecaster is None:
            return denormalize_forecast(current, self.config)

        # Until a full window has been seen, pad the start with the oldest reading.
        window = list(self.history)
        window = [window[0]] * (self.sequence_length - len(window)) + window
        with torch.no_grad():
            sequence = torch.as_tensor(np.array(window), dtype=torch.float32).unsqueeze(0)
            change = self.forecaster(sequence).squeeze(0).numpy()
        return denormalize_forecast(current + change, self.config)
