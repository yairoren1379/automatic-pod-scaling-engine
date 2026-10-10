import numpy as np
import pytest

from core.state import STATE_SIZE, ClusterMetrics
from predictor.lstm_forecaster import LoadForecaster
from predictor.predictor import Predictor
from predictor.smoothing import HysteresisRegime, MovingAverage


def test_moving_average_filters_a_single_spike():
    average = MovingAverage(window=3)
    for value in (50, 50, 90):
        smoothed = average.update([value])
    assert smoothed[0] == pytest.approx(63.33, abs=0.01)


def test_hysteresis_does_not_flicker_between_thresholds():
    regime = HysteresisRegime(high_enter=75, high_exit=60, low_enter=25, low_exit=35)
    readings = [70, 76, 72, 66, 74, 61, 59, 30, 24, 30, 34, 36]
    states = [regime.update(cpu) for cpu in readings]
    #           70 76 72 66 74 61 59 30 24  30  34  36
    assert states == [0, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, 0]


def test_hysteresis_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        HysteresisRegime(high_enter=60, high_exit=75, low_enter=25, low_exit=35)


def test_predictor_builds_full_state(config):
    reading = ClusterMetrics(cpu_percent=50, ram_percent=60, latency_ms=80, qps=400,
                             error_rate_percent=1, replicas=5)
    naive = Predictor(config).observe(reading)
    assert naive.state.shape == (STATE_SIZE,)
    assert naive.forecast["qps"] == pytest.approx(400, rel=1e-3)   # naive: future == now

    lstm = Predictor(config, LoadForecaster(hidden_size=8), {"sequence_length": 4})
    for _ in range(6):   # more readings than the window
        output = lstm.observe(reading)
    assert output.state.shape == (STATE_SIZE,)
    assert np.all(np.isfinite(output.state))
