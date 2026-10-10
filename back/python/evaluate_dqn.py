"""Compares a trained DQN against the threshold baseline in the mock environment.

Both controllers face exactly the same load: for every test episode the environment is created
with the same seed, so the load profile is identical. The DQN acts greedily (epsilon = 0).

Usage, from back/python:
    python evaluate_dqn.py --run-name decay_20261008_120000

Prints a table per load profile and saves it to runs/<run-name>/evaluation.json.
"""
import argparse
import json
import os

import numpy as np
import torch

from agents.dqn.dqn_agent import DQNAgent
from core.actions import NUM_ACTIONS, SCALE_UP, SCALE_DOWN
from core.state import STATE_SIZE
from predictor.predictor import Predictor
from simulation.load_profiles import PROFILE_KINDS
from simulation.mock_env import MockKubernetesEnv
from simulation.threshold_policy import threshold_policy

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

# A scale action in the opposite direction of one taken this recently counts as a flap.
FLAP_WINDOW_STEPS = 5


class DQNPolicy:
    """The trained agent, seeing states built by the Predictor (as in production)."""

    def __init__(self, agent, predictor):
        self.agent, self.predictor = agent, predictor

    def reset(self):
        self.predictor.reset()

    def act(self, metrics):
        return self.agent.select_action(self.predictor.observe(metrics).state, epsilon=0.0)


class ThresholdPolicy:
    """The baseline reacts to the raw CPU reading, like the Go controller's baseline mode."""

    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    def act(self, metrics):
        return threshold_policy(metrics, self.config)


def run_episode(env, policy, config):
    """Plays one episode with the given policy and returns its stats."""
    policy.reset()
    metrics = env.reset()
    slo = config["reward"]["latency_slo_ms"]
    total_reward, severe_failures, slo_violations, blocked = 0.0, 0, 0, 0
    latencies, replicas = [], []
    scale_actions, flaps = 0, 0
    last_scale_action, last_scale_step = None, None

    for step in range(env.episode_steps):
        action = policy.act(metrics)
        metrics, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        severe_failures += info["reward"]["severe_failure"]
        slo_violations += metrics.latency_ms > slo
        latencies.append(metrics.latency_ms)
        replicas.append(metrics.replicas)
        blocked += info["blocked"]

        if action in (SCALE_UP, SCALE_DOWN) and not info["blocked"]:
            scale_actions += 1
            reversed_direction = last_scale_action is not None and action != last_scale_action
            if reversed_direction and step - last_scale_step <= FLAP_WINDOW_STEPS:
                flaps += 1
            last_scale_action, last_scale_step = action, step

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "severe_failures": severe_failures,
        "slo_violation_percent": 100.0 * slo_violations / len(latencies),
        "avg_latency_ms": float(np.mean(latencies)),
        "avg_replicas": float(np.mean(replicas)),
        "scale_actions": scale_actions,
        "flaps": flaps,
        "blocked_actions": blocked,
    }


def load_run_policy(run_dir):
    """Loads a training run's config, agent and forecaster. Returns (config, DQNPolicy)."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)["config"]
    agent = DQNAgent(STATE_SIZE, NUM_ACTIONS, config["dqn"])
    agent.load(os.path.join(run_dir, "dqn.pt"))
    lstm_path = os.path.join(run_dir, "lstm.pt")
    predictor = Predictor.from_checkpoint(config, lstm_path if os.path.exists(lstm_path) else None)
    return config, DQNPolicy(agent, predictor)


def evaluate(config, dqn_policy, episodes_per_profile, seed):
    policies = {"dqn": dqn_policy, "threshold": ThresholdPolicy(config)}
    results = {}
    for profile in PROFILE_KINDS:
        results[profile] = {}
        for name, policy in policies.items():
            episodes = [
                run_episode(MockKubernetesEnv(config, seed=seed + i, profile_kind=profile), policy, config)
                for i in range(episodes_per_profile)
            ]
            # Average every stat over the test episodes.
            results[profile][name] = {key: float(np.mean([e[key] for e in episodes])) for key in episodes[0]}
    return results


def print_table(results):
    columns = [
        ("total_reward", "reward", "{:9.1f}"),
        ("severe_failures", "failures", "{:8.2f}"),
        ("slo_violation_percent", "SLO miss %", "{:10.1f}"),
        ("avg_latency_ms", "latency", "{:7.0f}"),
        ("avg_replicas", "replicas", "{:8.2f}"),
        ("scale_actions", "scalings", "{:8.1f}"),
        ("flaps", "flaps", "{:5.1f}"),
        ("blocked_actions", "blocked", "{:7.1f}"),
    ]
    header = f"{'profile':9} {'policy':9} " + " ".join(f"{label:>{len(fmt.format(0))}}" for _, label, fmt in columns)
    print(header)
    print("-" * len(header))
    for profile, by_policy in results.items():
        for name, stats in by_policy.items():
            print(f"{profile:9} {name:9} " + " ".join(fmt.format(stats[key]) for key, _, fmt in columns))
    print("\nAll values are per-episode averages. Lower is better except reward.")


def main():
    parser = argparse.ArgumentParser(description="Compare a trained DQN with the threshold baseline.")
    parser.add_argument("--run-name", required=True, help="folder under runs/ with dqn.pt and config.json")
    parser.add_argument("--episodes-per-profile", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000, help="first test seed (training uses low seeds)")
    parser.add_argument("--runs-dir", default=RUNS_DIR, help="parent folder of the run folders")
    args = parser.parse_args()

    torch.set_num_threads(1)
    run_dir = os.path.join(args.runs_dir, args.run_name)
    # Evaluate with the exact configuration and forecaster the model was trained with.
    config, dqn_policy = load_run_policy(run_dir)

    results = evaluate(config, dqn_policy, args.episodes_per_profile, args.seed)
    print_table(results)

    with open(os.path.join(run_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
