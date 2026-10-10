"""Runs the real-cluster stages of the sim-to-real pipeline and records their results.

    baseline  the threshold controller under every load profile (the comparison numbers)
    shadow    the threshold controller drives, the DQN fine-tunes on real transitions
    active    the DQN drives through the guardrails

Every load profile starts from the same state (load off, a fixed replica count), so stages are
compared fairly. The script only talks to the API server (mode, load, config, history) and
kubectl (to reset the replica count); the controller in the cluster does the rest.

Usage, from back/python (with the cluster and the API server running):
    python run_cluster_stages.py --stage baseline
    python run_cluster_stages.py --stage shadow
    python run_cluster_stages.py --stage active
    python run_cluster_stages.py --compare runs/cluster/baseline_... runs/cluster/active_... --plot cmp.png

Output: runs/cluster/<stage>_<time>/ with points.csv (every control loop) and summary.json.
"""
import argparse
import csv
import json
import os
import subprocess
import time
import urllib.request

from core.actions import SCALE_DOWN, SCALE_UP, action_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "cluster")
PROFILES = ["constant", "spike", "periodic"]
FLAP_WINDOW_STEPS = 5


def api(method, path, body=None, base="http://127.0.0.1:8000"):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def reset_cluster(replicas, settle_seconds):
    """Same starting point for every profile: idle traffic and a fixed replica count."""
    api("POST", "/api/load", {"profile": "off"})
    subprocess.run(["kubectl", "scale", "deployment/yair-api", f"--replicas={replicas}"], check=True,
                   capture_output=True)
    subprocess.run(["kubectl", "rollout", "status", "deployment/yair-api", "--timeout=180s"], check=True,
                   capture_output=True)
    time.sleep(settle_seconds)


def summarize(points, config):
    """The same statistics as evaluate_dqn.py, computed from the recorded control loops."""
    reward_cfg = config["reward"]
    severe = sum(1 for p in points if p["pod_crashed"] or p["latency_ms"] >= reward_cfg["extreme_latency_ms"]
                 or p["error_rate_percent"] >= reward_cfg["error_spike_percent"])
    scale_actions, flaps, blocked = 0, 0, 0
    last_action, last_step = None, None
    for step, p in enumerate(points):
        blocked += bool(p["blocked"])
        action = action_id(p["executed_action"])
        if action in (SCALE_UP, SCALE_DOWN) and not p["blocked"]:
            scale_actions += 1
            if last_action is not None and action != last_action and step - last_step <= FLAP_WINDOW_STEPS:
                flaps += 1
            last_action, last_step = action, step
    rewards = [p["reward"] for p in points if p["reward"] is not None]
    n = max(1, len(points))
    return {
        "control_loops": len(points),
        "total_reward": round(sum(rewards), 2),
        "severe_failures": severe,
        "slo_violation_percent": round(100 * sum(p["latency_ms"] > reward_cfg["latency_slo_ms"] for p in points) / n, 1),
        "avg_latency_ms": round(sum(p["latency_ms"] for p in points) / n, 1),
        "avg_error_rate_percent": round(sum(p["error_rate_percent"] for p in points) / n, 2),
        "avg_cpu_percent": round(sum(p["cpu_percent"] for p in points) / n, 1),
        "avg_replicas": round(sum(p["replicas"] for p in points) / n, 2),
        "max_replicas": max((p["replicas"] for p in points), default=0),
        "scale_actions": scale_actions,
        "flaps": flaps,
        "blocked_actions": blocked,
    }


def run_stage(args):
    config = api("GET", "/api/config")
    # Profile timing for the cluster runs: short enough to fit a run, long enough to scale.
    api("PUT", "/api/config", {"load_profiles": {
        "spike_after_seconds": 120, "spike_duration_seconds": args.spike_seconds,
        "period_seconds": args.period_seconds, "spike_multiplier": 2.0, "amplitude": 0.5}})
    api("POST", "/api/mode", {"mode": args.stage})

    run_dir = os.path.join(OUTPUT_DIR, f"{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    summary = {"stage": args.stage, "base_qps": args.base_qps, "minutes_per_profile": args.minutes,
               "start_replicas": args.start_replicas,
               "latency_slo_ms": config["reward"]["latency_slo_ms"], "profiles": {}}
    all_points = []
    try:
        for profile in args.profiles.split(","):
            print(f"[{args.stage}] {profile}: resetting the cluster", flush=True)
            # Start the load right after the reset, before the controller can scale away from it.
            reset_cluster(args.start_replicas, settle_seconds=5)
            api("POST", "/api/load", {"profile": profile, "base_qps": args.base_qps})
            start = time.time()
            print(f"[{args.stage}] {profile}: running for {args.minutes} minutes", flush=True)
            time.sleep(args.minutes * 60)
            points = [p for p in api("GET", "/api/history?limit=720") if p["timestamp"] >= start]
            for p in points:
                p["profile"] = profile
            all_points += points
            summary["profiles"][profile] = summarize(points, config)
            print(f"[{args.stage}] {profile}: {json.dumps(summary['profiles'][profile])}", flush=True)
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
    finally:
        api("POST", "/api/load", {"profile": "off"})
        api("PUT", "/api/config", {"load_profiles": config["load_profiles"]})

    if all_points:
        with open(os.path.join(run_dir, "points.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_points[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_points)
    print(f"Saved {run_dir}", flush=True)


def load_points(run_dir):
    with open(os.path.join(run_dir, "points.csv"), newline="") as f:
        return list(csv.DictReader(f))


def plot_comparison(run_dirs, summaries, path):
    """One row per load profile: requests/s, replicas and latency over time, one line per stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    # Shadow is the threshold policy again, so it gets a dashed line: identity is not color alone.
    styles = {"shadow": "--"}
    labels = {"baseline": "baseline: thresholds drive",
              "shadow": "shadow: thresholds drive, DQN learns",
              "active": "active: DQN drives"}
    slo = summaries[0].get("latency_slo_ms", 200)
    fig, axes = plt.subplots(len(PROFILES), 3, figsize=(14, 3.2 * len(PROFILES)), squeeze=False)
    for col, title in enumerate(["Requests per second", "Replicas", "Latency (ms, log scale)"]):
        axes[0][col].set_title(title, loc="left", fontsize=10)
    for run_dir, summary, color in zip(run_dirs, summaries, colors):
        points = load_points(run_dir)
        stage = summary["stage"]
        for row, profile in enumerate(PROFILES):
            rows = [p for p in points if p["profile"] == profile]
            if not rows:
                continue
            t0 = float(rows[0]["timestamp"])
            minutes = [(float(p["timestamp"]) - t0) / 60 for p in rows]
            for col, key in enumerate(["qps", "replicas", "latency_ms"]):
                ax = axes[row][col]
                ax.plot(minutes, [float(p[key]) for p in rows], color=color, linewidth=2,
                        linestyle=styles.get(stage, "-"), label=labels.get(stage, stage),
                        drawstyle="steps-post" if key == "replicas" else "default")
    for row, profile in enumerate(PROFILES):
        axes[row][0].set_ylabel(profile, fontsize=10)
        latency_ax = axes[row][2]
        latency_ax.set_yscale("log")
        # Plain numbers (50, 100, 200 ...) instead of matplotlib's 2x10^2 labels.
        latency_ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
        latency_ax.yaxis.set_major_formatter(ScalarFormatter())
        latency_ax.yaxis.set_minor_locator(NullLocator())
        latency_ax.yaxis.set_minor_formatter(NullFormatter())
        latency_ax.axhline(slo, color="#52514e", linewidth=1)
        latency_ax.set_ylim(bottom=min(20, latency_ax.get_ylim()[0]))
        # Label at the right end, where every run is back to low latency (no overlap).
        latency_ax.text(0.99, slo * 1.08, f"SLO {slo} ms", color="#52514e", fontsize=8, ha="right",
                        transform=latency_ax.get_yaxis_transform())
        for ax in axes[row]:
            ax.grid(color="#e6e5e0", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors="#52514e", labelsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Minutes since the load started")
    handles, legend_labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Real cluster: threshold baseline vs DQN, same load profiles", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare(run_dirs, plot_path=None):
    summaries = []
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "summary.json")) as f:
            summaries.append(json.load(f))
    if plot_path:
        plot_comparison(run_dirs, summaries, plot_path)
    keys = [("severe_failures", "failures"), ("slo_violation_percent", "SLO miss %"),
            ("avg_latency_ms", "latency"), ("avg_error_rate_percent", "errors %"),
            ("avg_replicas", "replicas"), ("total_reward", "reward"), ("scale_actions", "scalings"),
            ("flaps", "flaps"), ("blocked_actions", "blocked")]
    print(f"{'profile':9} {'stage':9} " + " ".join(f"{label:>10}" for _, label in keys))
    for profile in PROFILES:
        for s in summaries:
            if profile in s["profiles"]:
                row = s["profiles"][profile]
                print(f"{profile:9} {s['stage']:9} " + " ".join(f"{row[k]:>10}" for k, _ in keys))


def main():
    parser = argparse.ArgumentParser(description="Run the baseline / shadow / active stages on the cluster.")
    parser.add_argument("--stage", choices=["baseline", "shadow", "active"])
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--minutes", type=float, default=12, help="minutes per load profile")
    parser.add_argument("--base-qps", type=float, default=300)
    parser.add_argument("--start-replicas", type=int, default=3)
    parser.add_argument("--spike-seconds", type=float, default=360)
    parser.add_argument("--period-seconds", type=float, default=600)
    parser.add_argument("--compare", nargs="+", help="run folders to compare")
    parser.add_argument("--plot", help="with --compare: save a comparison figure to this path")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.plot)
    elif args.stage:
        run_stage(args)
    else:
        parser.error("give --stage or --compare")


if __name__ == "__main__":
    main()
