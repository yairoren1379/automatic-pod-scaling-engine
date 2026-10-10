"""Runs the proposal's experiments and summarizes them.

    1. Grid of learning rate x gamma (decaying epsilon, with the LSTM).
    2. Constant vs decaying epsilon (at the default learning rate and gamma).
    3. LSTM forecasts vs naive forecasts (at the default learning rate and gamma).

Every configuration is trained with several seeds (different random starts) so a result is a
mean +- standard deviation, not one lucky run. Each trained agent is then evaluated against the
threshold baseline on the same test loads.

Usage, from back/python (train the LSTM first):
    python run_experiments.py                  # everything, 3 seeds, in parallel
    python run_experiments.py --episodes 500   # quicker
    python run_experiments.py --summarize-only # rebuild the summary from finished runs

Output in runs/experiments/: one folder per run, summary.csv, summary.png, epsilon_comparison.png.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config_loader import APP_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "runs", "experiments")

LEARNING_RATES = [0.001, 0.0005, 0.00025]
GAMMAS = [0.95, 0.99]


def experiment_configs():
    """name -> extra train_dqn.py arguments."""
    configs = {}
    for lr in LEARNING_RATES:
        for gamma in GAMMAS:
            configs[f"lr{lr}_g{gamma}"] = ["--learning-rate", str(lr), "--gamma", str(gamma)]
    default = ["--learning-rate", str(APP_CONFIG["dqn"]["learning_rate"]), "--gamma", str(APP_CONFIG["dqn"]["gamma"])]
    configs["constant_epsilon"] = default + ["--epsilon-mode", "constant"]
    configs["no_lstm"] = default + ["--no-lstm"]
    return configs


def default_config_name():
    return f"lr{APP_CONFIG['dqn']['learning_rate']}_g{APP_CONFIG['dqn']['gamma']}"


def run(command):
    result = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(command)}\n{result.stderr[-2000:]}")
    return result.returncode


def train_and_evaluate(name, extra_args, seed, episodes, eval_episodes):
    run_name = f"{name}_s{seed}"
    start = time.time()
    train = [sys.executable, "train_dqn.py", "--run-name", run_name, "--runs-dir", EXPERIMENTS_DIR,
             "--seed", str(seed), "--no-replay-save", *extra_args]
    if episodes:
        train += ["--episodes", str(episodes)]
    if run(train) == 0:
        run([sys.executable, "evaluate_dqn.py", "--run-name", run_name, "--runs-dir", EXPERIMENTS_DIR,
             "--episodes-per-profile", str(eval_episodes)])
    print(f"  finished {run_name} in {time.time() - start:.0f}s")


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------

def load_evaluation(run_dir):
    """Per-run stats averaged over the three load profiles: {"dqn": {...}, "threshold": {...}}."""
    path = os.path.join(run_dir, "evaluation.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        by_profile = json.load(f)
    return {policy: {key: float(np.mean([by_profile[p][policy][key] for p in by_profile]))
                     for key in next(iter(by_profile.values()))[policy]}
            for policy in ("dqn", "threshold")}


def summarize(configs, seeds):
    rows, baseline = [], []
    for name in configs:
        evaluations = [load_evaluation(os.path.join(EXPERIMENTS_DIR, f"{name}_s{seed}")) for seed in seeds]
        evaluations = [e for e in evaluations if e]
        if not evaluations:
            continue
        baseline += [e["threshold"] for e in evaluations]
        row = {"config": name, "seeds": len(evaluations)}
        for key in ("total_reward", "severe_failures", "avg_replicas", "flaps", "slo_violation_percent",
                    "blocked_actions"):
            values = [e["dqn"][key] for e in evaluations]
            row[f"{key}_mean"] = round(float(np.mean(values)), 2)
            row[f"{key}_std"] = round(float(np.std(values)), 2)
        rows.append(row)

    if not rows:
        print("No finished runs to summarize.")
        return
    threshold = {key: round(float(np.mean([b[key] for b in baseline])), 2) for key in baseline[0]}

    with open(os.path.join(EXPERIMENTS_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(EXPERIMENTS_DIR, "threshold_baseline.json"), "w") as f:
        json.dump(threshold, f, indent=2)

    print(f"\n{'config':22} {'reward':>16} {'failures':>13} {'replicas':>13} {'flaps':>11}")
    for row in rows:
        print(f"{row['config']:22} {row['total_reward_mean']:8.1f} ±{row['total_reward_std']:6.1f} "
              f"{row['severe_failures_mean']:6.2f} ±{row['severe_failures_std']:4.2f} "
              f"{row['avg_replicas_mean']:6.2f} ±{row['avg_replicas_std']:4.2f} "
              f"{row['flaps_mean']:5.1f} ±{row['flaps_std']:3.1f}")
    print(f"{'threshold baseline':22} {threshold['total_reward']:8.1f}         "
          f"{threshold['severe_failures']:6.2f}        {threshold['avg_replicas']:6.2f}        "
          f"{threshold['flaps']:5.1f}")
    print("Per-episode averages over the three load profiles; ± is the standard deviation over seeds.")

    plot_summary(rows, threshold, os.path.join(EXPERIMENTS_DIR, "summary.png"))
    plot_epsilon_comparison(seeds, os.path.join(EXPERIMENTS_DIR, "epsilon_comparison.png"))


def _style(ax):
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbcb6")
    ax.tick_params(colors="#52514e", labelsize=9)


def plot_summary(rows, threshold, path):
    """Two small panels (reward, failures): one bar per configuration, baseline as a reference line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["config"] for row in rows]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(names) + 1.8), sharey=True)
    panels = [("total_reward", "Total reward per episode (higher is better)"),
              ("severe_failures", "Severe failures per episode (lower is better)")]
    for ax, (key, title) in zip(axes, panels):
        means = [row[f"{key}_mean"] for row in rows]
        stds = [row[f"{key}_std"] for row in rows]
        ax.barh(y, means, xerr=stds, height=0.55, color="#2a78d6",
                error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 3})
        ax.axvline(threshold[key], color="#52514e", linewidth=1.5)
        ax.text(threshold[key], len(names) - 0.4, " threshold baseline", color="#52514e", fontsize=8,
                va="bottom")
        ax.set_title(title, loc="left", fontsize=10)
        _style(ax)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    fig.suptitle("DQN configurations vs the threshold baseline (mean ± std over seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_epsilon_comparison(seeds, path):
    """Learning curves of decaying vs constant epsilon, averaged over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {"decaying epsilon": default_config_name(), "constant epsilon": "constant_epsilon"}
    colors = {"decaying epsilon": "#2a78d6", "constant epsilon": "#eb6834"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, name in curves.items():
        per_seed = []
        for seed in seeds:
            path_csv = os.path.join(EXPERIMENTS_DIR, f"{name}_s{seed}", "metrics.csv")
            if os.path.exists(path_csv):
                with open(path_csv) as f:
                    per_seed.append([float(r["total_reward"]) for r in csv.DictReader(f)])
        if not per_seed:
            continue
        length = min(map(len, per_seed))
        mean = np.mean([s[:length] for s in per_seed], axis=0)
        window = 50
        smooth = np.convolve(mean, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, length + 1), smooth, color=colors[label], linewidth=2, label=label)
    ax.set_title("Training reward (50-episode moving average, mean over seeds)", loc="left", fontsize=10)
    ax.set_xlabel("Episode")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run the proposal's experiments.")
    parser.add_argument("--episodes", type=int, help="training episodes per run (default: dqn.episodes)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=20, help="test episodes per load profile")
    parser.add_argument("--parallel", type=int, default=min(os.cpu_count() or 4, 20))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--only", help="comma-separated configurations to (re)train, e.g. no_lstm,constant_epsilon")
    args = parser.parse_args()

    configs = experiment_configs()
    seeds = list(range(args.seeds))
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    if not args.summarize_only:
        selected = args.only.split(",") if args.only else list(configs)
        unknown = [name for name in selected if name not in configs]
        if unknown:
            raise SystemExit(f"unknown configurations {unknown}; choose from {list(configs)}")
        jobs = [(name, configs[name], seed) for name in selected for seed in seeds]
        print(f"Running {len(jobs)} training runs, {args.parallel} at a time...")
        start = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for name, extra, seed in jobs:
                pool.submit(train_and_evaluate, name, extra, seed, args.episodes, args.eval_episodes)
        print(f"All runs finished in {time.time() - start:.0f}s")

    summarize(configs, seeds)


if __name__ == "__main__":
    main()
