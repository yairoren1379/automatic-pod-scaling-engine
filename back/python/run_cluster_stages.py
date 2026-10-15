"""Runs the real-cluster stages of the sim-to-real pipeline and records their results.

    baseline  the threshold controller under every load profile (the comparison numbers)
    shadow    the threshold controller drives, the DQN fine-tunes on real transitions
    active    the DQN drives through the guardrails

Every load profile starts from the same state (load off, a fixed replica count), so stages are
compared fairly. The script only talks to the API server (mode, load, config, history) and
kubectl (to reset the replica count); the controller in the cluster does the rest.

During a reset the cluster is switched to baseline mode, so the DQN never learns from the
transition that spans it: the harness, not the agent's action, changed the cluster there.

Usage, from back/python (with the cluster and the API server running):
    python run_cluster_stages.py --stage baseline
    python run_cluster_stages.py --stage shadow
    python run_cluster_stages.py --stage active
    python run_cluster_stages.py --compare runs/cluster/baseline_... runs/cluster/active_... --plot cmp.png
    # a long shadow run: every profile at three load levels, twice (about 4 hours)
    python run_cluster_stages.py --stage shadow --qps-levels 200,300,400 --rounds 2
    python run_cluster_stages.py --plot-learning learning.png    # the online learning curve

Output: runs/cluster/<stage>_<time>/ with points.csv (every control loop) and summary.json.
"""
import argparse
import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from core.actions import SCALE_DOWN, SCALE_UP, action_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "cluster")
LEARNING_CSV = os.path.join(BASE_DIR, "data", "online_learning.csv")
PROFILES = ["constant", "spike", "periodic"]
FLAP_WINDOW_STEPS = 5


def api(method, path, body=None, base="http://127.0.0.1:8000", attempts=4):
    """One call to the API server. Retries a few times, so a brief hiccup (a slow response, the
    ZooKeeper port-forward restarting) does not end a run that has been going for hours."""
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError:
            raise                      # the API answered with an error: retrying won't help
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts:
                raise
            print(f"API {method} {path} failed ({error}); retrying in 10s", flush=True)
            time.sleep(10)


def keep_awake():
    """Asks Windows not to sleep while this script runs (released automatically when it exits).
    The screen may still turn off. Closing a laptop's lid can still suspend it."""
    if os.name == "nt":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def wait_for_decision(mode, since, timeout=150):
    """Waits until the controller has made a decision in `mode` after `since` (it decides every
    30 s). Only the controller's decisions set the mode in /api/status."""
    while time.time() - since < timeout:
        status = api("GET", "/api/status")
        if status["mode"] == mode and (status["last_contact"] or 0) > since:
            return
        time.sleep(3)
    raise RuntimeError(f"no decision in {mode} mode for {timeout}s: is the controller running?")


def pause_learning():
    """Baseline mode (no learning) until the controller has decided once in it. The transition
    that spans the following reset then starts in baseline mode and is not learned."""
    api("POST", "/api/mode", {"mode": "baseline"})
    wait_for_decision("baseline", since=time.time())


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
    keep_awake()
    config = api("GET", "/api/config")
    # Profile timing for the cluster runs: short enough to fit a run, long enough to scale.
    api("PUT", "/api/config", {"load_profiles": {
        "spike_after_seconds": 120, "spike_duration_seconds": args.spike_seconds,
        "period_seconds": args.period_seconds, "spike_multiplier": 2.0, "amplitude": 0.5}})
    learning = args.stage in ("shadow", "active")
    if not learning:
        api("POST", "/api/mode", {"mode": args.stage})

    # Segments: every profile at every load level, `rounds` times. A plain stage run is one
    # round at one level, and its segments are named after the profiles.
    levels = [float(q) for q in args.qps_levels.split(",")] if args.qps_levels else [args.base_qps]
    segments = []
    for round_number in range(1, args.rounds + 1):
        for qps in levels:
            for profile in args.profiles.split(","):
                name = profile + (f"@{qps:g}" if len(levels) > 1 else "") + \
                    (f"#{round_number}" if args.rounds > 1 else "")
                segments.append((name, profile, qps))

    run_dir = os.path.join(OUTPUT_DIR, f"{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    online_steps_start = api("GET", "/api/status")["online_steps"]
    summary = {"stage": args.stage, "base_qps": args.base_qps, "qps_levels": levels, "rounds": args.rounds,
               "minutes_per_profile": args.minutes, "start_replicas": args.start_replicas,
               "latency_slo_ms": config["reward"]["latency_slo_ms"],
               "online_steps_start": online_steps_start, "profiles": {}}
    all_points = []
    try:
        for number, (name, profile, qps) in enumerate(segments, start=1):
            print(f"[{args.stage}] {name} ({number}/{len(segments)}): resetting the cluster", flush=True)
            if learning:
                pause_learning()
            # Start the load right after the reset, before the controller can scale away from it.
            reset_cluster(args.start_replicas, settle_seconds=5)
            api("POST", "/api/load", {"profile": profile, "base_qps": qps})
            if learning:
                api("POST", "/api/mode", {"mode": args.stage})
            start = time.time()
            print(f"[{args.stage}] {name}: running for {args.minutes} minutes", flush=True)
            time.sleep(args.minutes * 60)
            points = [p for p in api("GET", "/api/history?limit=720") if p["timestamp"] >= start]
            for p in points:
                p.update({"profile": profile, "segment": name, "base_qps": qps})
            all_points += points
            summary["profiles"][name] = summarize(points, config)
            summary["online_steps_end"] = api("GET", "/api/status")["online_steps"]
            print(f"[{args.stage}] {name}: {json.dumps(summary['profiles'][name])}", flush=True)
            # Saved after every segment, so an interrupted run keeps what it recorded.
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            write_points(run_dir, all_points)
    finally:
        api("POST", "/api/load", {"profile": "off"})
        api("PUT", "/api/config", {"load_profiles": config["load_profiles"]})
    print(f"Saved {run_dir}", flush=True)


def write_points(run_dir, points):
    if points:
        with open(os.path.join(run_dir, "points.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(points[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(points)


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


def plot_learning_curve(path, csv_path=LEARNING_CSV, window=20):
    """The online learning curve on the real cluster, one point per learned transition: the TD
    loss (is the network still being surprised?) and the reward of the executed action."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["loss"]]   # baseline-mode rows have no loss
    if not rows:
        raise SystemExit(f"no learned transitions in {csv_path} yet")
    steps = [int(float(r["online_steps"])) for r in rows]

    def rolling_mean(values):
        return [sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1])
                for i in range(len(values))]

    colors = {"shadow": "#eb6834", "active": "#1baf7a"}   # same colors as the comparison figure
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    panels = [("loss", "TD loss (Huber), log scale", True),
              ("reward", "Reward of the executed action (shadow: thresholds, active: DQN)", False)]
    for ax, (key, title, log_scale) in zip(axes, panels):
        values = [float(r[key]) for r in rows]
        for mode, color in colors.items():
            mode_steps = [s for s, r in zip(steps, rows) if r["mode"] == mode]
            mode_values = [v for v, r in zip(values, rows) if r["mode"] == mode]
            if mode_steps:
                ax.scatter(mode_steps, mode_values, s=8, color=color, alpha=0.35, label=f"{mode} (one transition)")
        ax.plot(steps, rolling_mean(values), color="#2b2a27", linewidth=2, label=f"mean of the last {window}")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    axes[-1].set_xlabel("Real transitions learned (one per 30-second control loop)")
    fig.suptitle("Online learning on the real cluster", fontsize=12)
    fig.tight_layout()
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
    parser.add_argument("--qps-levels", help="comma list of base rates to cycle through, e.g. 200,300,400")
    parser.add_argument("--rounds", type=int, default=1, help="repeat all segments (long shadow runs)")
    parser.add_argument("--start-replicas", type=int, default=4)
    parser.add_argument("--spike-seconds", type=float, default=360)
    parser.add_argument("--period-seconds", type=float, default=600)
    parser.add_argument("--compare", nargs="+", help="run folders to compare")
    parser.add_argument("--plot", help="with --compare: save a comparison figure to this path")
    parser.add_argument("--plot-learning", help="save the online learning curve (data/online_learning.csv)")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.plot)
    elif args.plot_learning:
        plot_learning_curve(args.plot_learning)
    elif args.stage:
        run_stage(args)
    else:
        parser.error("give --stage, --compare or --plot-learning")


if __name__ == "__main__":
    main()
