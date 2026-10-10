"""Noise filters that run before the LSTM: moving average and hysteresis."""
from collections import deque

import numpy as np


class MovingAverage:
    """The average of the last `window` readings.

    A single noisy reading (one slow request batch, a GC pause) should not make the agent scale.
    Averaging a few readings keeps real trends and cancels momentary noise.
    """

    def __init__(self, window: int):
        self.readings = deque(maxlen=window)

    def update(self, values) -> np.ndarray:
        self.readings.append(np.asarray(values, dtype=np.float64))
        return np.mean(self.readings, axis=0)

    def reset(self):
        self.readings.clear()


class HysteresisRegime:
    """A load regime with two thresholds per direction (a Schmitt trigger).

    With a single threshold at 75% CPU, a load hovering around 75% would flip the signal every
    reading - the same flapping a threshold autoscaler suffers from. Here the regime enters "high"
    only above `high_enter` but leaves it only below the lower `high_exit`, so a value bouncing
    between the two thresholds keeps the previous regime. The same holds for "low".
    """
    LOW, NORMAL, HIGH = -1, 0, 1

    def __init__(self, high_enter: float, high_exit: float, low_enter: float, low_exit: float):
        if not (low_enter < low_exit <= high_exit < high_enter):
            raise ValueError("expected low_enter < low_exit <= high_exit < high_enter")
        self.high_enter, self.high_exit = high_enter, high_exit
        self.low_enter, self.low_exit = low_enter, low_exit
        self.regime = self.NORMAL

    def update(self, cpu_percent: float) -> int:
        if self.regime == self.HIGH and cpu_percent <= self.high_exit:
            self.regime = self.NORMAL
        elif self.regime == self.LOW and cpu_percent >= self.low_exit:
            self.regime = self.NORMAL

        if self.regime == self.NORMAL:
            if cpu_percent >= self.high_enter:
                self.regime = self.HIGH
            elif cpu_percent <= self.low_enter:
                self.regime = self.LOW
        return self.regime

    def reset(self):
        self.regime = self.NORMAL
