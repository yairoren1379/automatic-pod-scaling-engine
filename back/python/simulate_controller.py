"""A stand-in for the Go controller that drives the mock environment instead of Kubernetes.

It speaks to the API server exactly like the Go controller does (decide -> guardrails ->
execute -> report), so the whole system - learning engine, modes, dashboard, load profiles -
can be run and demonstrated on a laptop without a cluster.

Usage, from back/python (with the API server running):
    python simulate_controller.py                  # one control loop per second
    python simulate_controller.py --interval 0.2   # faster
"""
import argparse
import json
import time
import urllib.request
from dataclasses import asdict

from config_loader import APP_CONFIG
from core.actions import ACTION_NAMES, action_id
from core.state import ClusterMetrics
from simulation.load_profiles import profile_from_load_spec
from simulation.mock_env import MockKubernetesEnv
from simulation.threshold_policy import threshold_policy


def http_get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def http_post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def apply_guardrail_limits(env, config):
    """Live config changes (from the dashboard) reach the simulated guardrails too."""
    limits = config["guardrails"]
    env.min_replicas, env.max_replicas = limits["min_replicas"], limits["max_replicas"]
    env.stabilization_steps = round(limits["scale_down_stabilization_seconds"]
                                    / config["controller"]["loop_delay_seconds"])


def main():
    parser = argparse.ArgumentParser(description="Drive the API with the mock environment.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--interval", type=float, default=1.0, help="real seconds per control loop")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = MockKubernetesEnv(APP_CONFIG, seed=args.seed)
    env.reset()
    step_seconds = APP_CONFIG["controller"]["loop_delay_seconds"]   # what one step represents
    current_load = None
    print(f"Simulating the cluster against {args.api} (Ctrl+C to stop)")

    while True:
        loop_start = time.time()
        try:
            config = http_get(f"{args.api}/api/config")
            load = http_get(f"{args.api}/api/load")
        except OSError as e:
            print(f"API not reachable ({e}), retrying...")
            time.sleep(2)
            continue

        env.config = config   # the reward follows live config changes
        apply_guardrail_limits(env, config)
        if load != current_load:
            env.set_profile(profile_from_load_spec(load, config["load_profiles"], step_seconds))
            current_load = load
            print(f"Load profile: {load.get('profile')}")

        mode = config["controller"]["mode"]
        metrics: ClusterMetrics = env.metrics
        try:
            decision = http_post(f"{args.api}/api/decide", {"mode": mode, "metrics": asdict(metrics)})
        except OSError as e:
            decision = None
            print(f"decide failed ({e}): falling back to the baseline")

        # Who chooses: the DQN only in active mode (and only if the engine answered).
        if mode == "active" and decision:
            action, source = action_id(decision["action"]), "dqn"
        else:
            action, source = threshold_policy(metrics, config), "baseline"

        replicas_before = env.replicas
        _, _, _, _, info = env.step(action)   # the environment applies the Go guardrails' rules
        reason = info["block_reason"]

        if decision:
            http_post(f"{args.api}/api/report", {
                "decision_id": decision["decision_id"], "executed_action": ACTION_NAMES[action],
                "source": source, "blocked": bool(reason), "block_reason": reason,
                "replicas_before": replicas_before, "replicas_after": env.replicas,
            })
        m = env.metrics
        print(f"[{mode:8}] {ACTION_NAMES[action]:9} ({source:8}) replicas={m.replicas:2d} "
              f"cpu={m.cpu_percent:5.1f}% lat={m.latency_ms:6.0f}ms qps={m.qps:6.0f} err={m.error_rate_percent:4.1f}%"
              + (f"  BLOCKED: {reason}" if reason else ""))
        time.sleep(max(0.0, args.interval - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
