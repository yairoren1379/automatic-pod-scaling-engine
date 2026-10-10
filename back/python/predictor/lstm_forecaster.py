"""The LSTM that forecasts CPU, RAM, latency and QPS a few steps ahead."""
import torch
from torch import nn

from core.state import FORECAST_METRICS, METRICS_VECTOR_SIZE


class LoadForecaster(nn.Module):
    """Reads the last `sequence_length` metric vectors and predicts how much each forecast
    metric will change `horizon` steps from now.

    Why an LSTM: the answer depends on the trend, not on one reading. QPS that has been rising
    for five minutes will probably keep rising; the LSTM's memory cell carries that trend from
    one time step to the next.

    Why predict the change and not the value: "no change" is already a decent guess, so the
    network only has to learn the correction. That is easier to learn and to evaluate - the
    naive forecast ("the future equals now") is exactly a predicted change of zero.
    """

    def __init__(self, hidden_size: int, input_size: int = METRICS_VECTOR_SIZE,
                 output_size: int = len(FORECAST_METRICS)):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """sequences: (batch, sequence_length, input_size) -> predicted change (batch, output_size)."""
        outputs, _ = self.lstm(sequences)
        last_step = outputs[:, -1, :]   # the LSTM's summary after reading the whole window
        return self.head(last_step)


def save_forecaster(model: LoadForecaster, path: str, meta: dict):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_forecaster(path: str):
    """Returns (model, meta). meta holds sequence_length, horizon_steps and the accuracy report."""
    checkpoint = torch.load(path, weights_only=False)
    meta = checkpoint["meta"]
    model = LoadForecaster(hidden_size=meta["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, meta
