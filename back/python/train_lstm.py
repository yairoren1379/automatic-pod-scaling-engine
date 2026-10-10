"""Trains the LSTM load forecaster.

Usage, from back/python:
    python train_lstm.py                              # data from the mock environment
    python train_lstm.py --from-csv data/metrics.csv  # data recorded on the real cluster

Data from the mock environment is collected with the threshold controller plus 30% random
actions, so the history covers both well-managed and badly-managed clusters.

Saves models/lstm.pt and models/lstm_report.png (forecast vs reality on unseen episodes), and
prints the forecast error next to the naive forecast ("the future equals now") - the LSTM is
only useful if it beats that.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch import nn

from config_loader import APP_CONFIG
from core.state import FORECAST_METRICS, ClusterMetrics, normalize_metrics
from predictor.lstm_forecaster import LoadForecaster, save_forecaster
from predictor.predictor import Predictor
from simulation.mock_env import MockKubernetesEnv
from simulation.threshold_policy import threshold_policy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

EXPLORATION_RATE = 0.3   # share of random actions while collecting mock data


# ---------------------------------------------------------------------------------------------
# Collecting sequences of smoothed, normalized metric vectors
# ---------------------------------------------------------------------------------------------

def smoothed_vectors(metrics_sequence, config):
    """Runs the Predictor's moving average, exactly as at inference time."""
    predictor = Predictor(config)
    return np.array([normalize_metrics(predictor.observe(m).smoothed, config) for m in metrics_sequence])


def collect_mock_sequences(config, episodes, seed):
    rng = random.Random(seed)
    env = MockKubernetesEnv(config, seed=seed)
    sequences = []
    for _ in range(episodes):
        metrics = [env.reset()]
        while True:
            action = rng.randrange(3) if rng.random() < EXPLORATION_RATE else threshold_policy(env.metrics, config)
            next_metrics, _, terminated, truncated, _ = env.step(action)
            metrics.append(next_metrics)
            if terminated or truncated:
                break
        sequences.append(smoothed_vectors(metrics, config))
    return sequences


def collect_csv_sequences(path, config, max_gap_seconds):
    """Reads the metrics the API server records; splits wherever the recording had a gap."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    segments, current, last_time = [], [], None
    for row in rows:
        timestamp = float(row["timestamp"])
        if last_time is not None and timestamp - last_time > max_gap_seconds:
            segments.append(current)
            current = []
        current.append(ClusterMetrics(
            cpu_percent=float(row["cpu_percent"]), ram_percent=float(row["ram_percent"]),
            latency_ms=float(row["latency_ms"]), qps=float(row["qps"]),
            error_rate_percent=float(row["error_rate_percent"]), replicas=int(float(row["replicas"])),
        ))
        last_time = timestamp
    segments.append(current)
    return [smoothed_vectors(segment, config) for segment in segments if len(segment) > 1]


def make_samples(sequences, sequence_length, horizon, time_ranges=None):
    """(window of the last `sequence_length` vectors, change of the forecast metrics `horizon`
    steps later). Windows at the start of a sequence are padded like the Predictor does.

    time_ranges: optional (first, last) window-end index per sequence, to take samples from only
    one part of each sequence (a chronological train/validation split)."""
    windows, targets = [], []
    n = len(FORECAST_METRICS)
    for i, seq in enumerate(sequences):
        first, last = time_ranges[i] if time_ranges else (0, len(seq) - horizon)
        for t in range(first, min(last, len(seq) - horizon)):
            start = max(0, t - sequence_length + 1)
            window = list(seq[start:t + 1])
            window = [window[0]] * (sequence_length - len(window)) + window
            windows.append(window)
            targets.append(seq[t + horizon][:n] - seq[t][:n])
    return torch.tensor(np.array(windows), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)


# ---------------------------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------------------------

def train(model, train_x, train_y, val_x, val_y, predictor_cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=predictor_cfg["learning_rate"])
    loss_fn = nn.MSELoss()
    batch_size = predictor_cfg["batch_size"]
    best_val, best_state = float("inf"), None

    for epoch in range(1, predictor_cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x))
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        print(f"Epoch {epoch:3d} | validation MSE {val_loss:.6f}")
        # Keep the epoch that did best on unseen data (early stopping without stopping).
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()


def rmse_report(model, val_x, val_y, config):
    """Forecast error per metric in real units, LSTM vs naive (predicted change = 0)."""
    with torch.no_grad():
        predicted = model(val_x).numpy()
    actual = val_y.numpy()
    scales = [100.0, 100.0, config["state_normalization"]["latency_scale_ms"],
              config["state_normalization"]["qps_scale"]]
    report = {}
    for i, name in enumerate(FORECAST_METRICS):
        lstm = float(np.sqrt(np.mean((predicted[:, i] - actual[:, i]) ** 2)) * scales[i])
        naive = float(np.sqrt(np.mean(actual[:, i] ** 2)) * scales[i])
        report[name] = {"lstm_rmse": round(lstm, 3), "naive_rmse": round(naive, 3),
                        "improvement_percent": round(100 * (1 - lstm / naive), 1) if naive > 0 else 0.0}
    return report


def plot_example(model, sequence, sequence_length, horizon, config, path):
    """Forecast vs reality on one unseen episode, for QPS and CPU."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows, _ = make_samples([sequence], sequence_length, horizon)
    with torch.no_grad():
        change = model(windows).numpy()
    n = len(sequence) - horizon
    steps = np.arange(horizon, horizon + n)
    qps_scale = config["state_normalization"]["qps_scale"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    panels = [("QPS (requests per second)", 3, qps_scale), ("CPU (%)", 0, 100.0)]
    for ax, (title, index, scale) in zip(axes, panels):
        actual = sequence[horizon:, index] * scale
        forecast = (sequence[:n, index] + change[:, index]) * scale
        ax.plot(steps, actual, color="#2a78d6", linewidth=2, label="actual")
        ax.plot(steps, forecast, color="#eb6834", linewidth=2, label=f"LSTM forecast ({horizon} steps earlier)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bdbcb6")
        ax.tick_params(colors="#52514e", labelsize=9)
    axes[-1].set_xlabel("Step (30 seconds each)")
    fig.suptitle("LSTM forecast vs reality on an unseen episode", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM load forecaster.")
    parser.add_argument("--from-csv", help="train on metrics recorded by the API server instead of the mock env")
    parser.add_argument("--episodes", type=int, help="override predictor.training_episodes (mock data)")
    parser.add_argument("--output", default=os.path.join(MODELS_DIR, "lstm.pt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = APP_CONFIG
    predictor_cfg = config["predictor"]
    sequence_length, horizon = predictor_cfg["sequence_length"], predictor_cfg["horizon_steps"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    start = time.time()
    if args.from_csv:
        max_gap = 3 * config["controller"]["loop_delay_seconds"]
        sequences = collect_csv_sequences(args.from_csv, config, max_gap)
        source = f"csv:{args.from_csv}"
    else:
        episodes = args.episodes or predictor_cfg["training_episodes"]
        sequences = collect_mock_sequences(config, episodes, args.seed)
        source = f"mock_env:{episodes}_episodes"
    print(f"Collected {len(sequences)} sequences ({sum(map(len, sequences))} readings) "
          f"in {time.time() - start:.0f}s from {source}")

    if len(sequences) >= 5:
        # Many episodes (mock data): split by episode, so validation episodes are unseen.
        random.Random(args.seed).shuffle(sequences)
        split = max(1, int(0.8 * len(sequences)))
        train_x, train_y = make_samples(sequences[:split], sequence_length, horizon)
        val_x, val_y = make_samples(sequences[split:], sequence_length, horizon)
        example = sequences[split]
    else:
        # A few long recordings (real cluster): train on the first 80% of each recording and
        # validate on the rest, i.e. forecast a future the model has not seen.
        cuts = [int(0.8 * len(seq)) for seq in sequences]
        train_x, train_y = make_samples(sequences, sequence_length, horizon,
                                        [(0, cut - horizon) for cut in cuts])
        val_x, val_y = make_samples(sequences, sequence_length, horizon,
                                    [(cut, len(seq)) for cut, seq in zip(cuts, sequences)])
        example = max(sequences, key=len)
    print(f"{len(train_x)} training samples, {len(val_x)} validation samples")
    if len(train_x) < 50 or len(val_x) < 10:
        raise SystemExit("Not enough data to train the LSTM: record more control loops first "
                         f"(have {sum(map(len, sequences))} readings).")

    model = LoadForecaster(hidden_size=predictor_cfg["hidden_size"])
    train(model, train_x, train_y, val_x, val_y, predictor_cfg)

    report = rmse_report(model, val_x, val_y, config)
    print(f"\nForecast error {horizon} steps ahead (RMSE, real units), validation set:")
    print(f"{'metric':12} {'LSTM':>10} {'naive':>10} {'better by':>10}")
    for name, row in report.items():
        print(f"{name:12} {row['lstm_rmse']:10.2f} {row['naive_rmse']:10.2f} {row['improvement_percent']:9.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_forecaster(model, args.output, meta={
        "sequence_length": sequence_length, "horizon_steps": horizon,
        "hidden_size": predictor_cfg["hidden_size"], "source": source, "validation_rmse": report,
    })
    report_png = os.path.splitext(args.output)[0] + "_report.png"
    plot_example(model, example, sequence_length, horizon, config, report_png)
    print(f"\nSaved {args.output} and {report_png} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
