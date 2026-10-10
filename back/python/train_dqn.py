"""Pre-trains the DQN agent in the mock environment (stage 1 of the sim-to-real pipeline).

Usage, from back/python (train the LSTM first: python train_lstm.py):
    python train_dqn.py                                  # decaying epsilon, config from ZooKeeper
    python train_dqn.py --epsilon-mode constant          # the constant-epsilon comparison run
    python train_dqn.py --learning-rate 0.0005 --gamma 0.99 --run-name lr5e-4_g099
    python train_dqn.py --no-lstm                        # naive forecasts, to measure the LSTM's value

Everything a run produces goes to runs/<run-name>/:
    config.json         the exact configuration used, for reproducibility
    metrics.csv         one row per episode: reward, loss, epsilon, failures, validation...
    dqn.pt              the checkpoint that did best on validation (networks + optimizer)
    dqn_last.pt         the checkpoint at the end of training
    lstm.pt             copy of the forecaster the agent was trained with
    replay_buffer.pkl   the replay memory, to resume training later
    learning_curve.png
"""
import argparse
import copy
import csv
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from config_loader import APP_CONFIG
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.epsilon import EpsilonSchedule
from core.actions import NUM_ACTIONS
from core.state import STATE_SIZE
from evaluate_dqn import DQNPolicy, run_episode
from predictor.predictor import Predictor
from simulation.load_profiles import PROFILE_KINDS
from simulation.mock_env import MockKubernetesEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
DEFAULT_LSTM = os.path.join(BASE_DIR, "models", "lstm.pt")

METRIC_FIELDS = [
    "episode", "profile", "total_reward", "avg_loss", "epsilon", "severe_failures",
    "blocked_actions", "avg_replicas", "avg_latency_ms", "total_steps", "validation_reward",
]

# Validation episodes use their own seeds: different from training (0, 1, 2...) and from the
# final evaluation in evaluate_dqn.py (10000+), so the model is not selected on its test set.
VALIDATION_SEED = 5000


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train the DQN autoscaler in the mock environment.")
    parser.add_argument("--episodes", type=int, help="override dqn.episodes")
    parser.add_argument("--epsilon-mode", choices=["decay", "constant"], default="decay")
    parser.add_argument("--learning-rate", type=float, help="override dqn.learning_rate")
    parser.add_argument("--gamma", type=float, help="override dqn.gamma")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", help="output folder under runs/ (default: mode + timestamp)")
    parser.add_argument("--runs-dir", default=RUNS_DIR, help="parent folder of the run folders")
    parser.add_argument("--lstm", default=DEFAULT_LSTM, help="forecaster checkpoint (train_lstm.py)")
    parser.add_argument("--no-lstm", action="store_true", help="use the naive forecast instead of the LSTM")
    parser.add_argument("--no-replay-save", action="store_true",
                        help="don't save the replay buffer (saves disk space in experiments)")
    return parser.parse_args()


def run_training_episode(env, predictor, agent, epsilon_schedule, total_steps):
    """Plays one episode, learning after every step. Returns the episode stats."""
    predictor.reset()
    state = predictor.observe(env.reset()).state
    total_reward, losses = 0.0, []
    severe_failures, blocked_actions = 0, 0
    replicas_sum, latency_sum, steps = 0, 0.0, 0

    while True:
        epsilon = epsilon_schedule.value(total_steps)
        action = agent.select_action(state, epsilon)
        metrics, reward, terminated, truncated, info = env.step(action)
        next_state = predictor.observe(metrics).state

        # Only a real terminal state cuts off the future. Hitting the episode's time limit
        # (truncated) does not: the cluster would keep running after it.
        agent.remember(state, action, reward, next_state, terminated)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        total_reward += reward
        severe_failures += info["reward"]["severe_failure"]
        blocked_actions += info["blocked"]
        replicas_sum += info["metrics"]["replicas"]
        latency_sum += info["metrics"]["latency_ms"]
        steps += 1
        total_steps += 1
        state = next_state

        if terminated or truncated:
            break

    return {
        "profile": env.profile.kind,
        "total_reward": round(total_reward, 2),
        "avg_loss": round(float(np.mean(losses)), 4) if losses else "",
        "epsilon": round(epsilon, 4),
        "severe_failures": severe_failures,
        "blocked_actions": blocked_actions,
        "avg_replicas": round(replicas_sum / steps, 2),
        "avg_latency_ms": round(latency_sum / steps, 1),
        "total_steps": total_steps,
    }


def validate(config, agent, predictor):
    """Average total reward of the greedy policy on fixed validation episodes."""
    policy = DQNPolicy(agent, predictor)
    per_profile = config["dqn"]["validation_episodes_per_profile"]
    rewards = [
        run_episode(MockKubernetesEnv(config, seed=VALIDATION_SEED + i, profile_kind=profile), policy, config)["total_reward"]
        for profile in PROFILE_KINDS for i in range(per_profile)
    ]
    return float(np.mean(rewards))


def plot_learning_curve(rows, path, title):
    """Four small panels, one measure each, so every measure keeps its own scale."""
    import matplotlib
    matplotlib.use("Agg")   # render to a file, no window
    import matplotlib.pyplot as plt

    color = "#2a78d6"
    episodes = [row["episode"] for row in rows]

    def moving_average(values, window=50):
        values = np.asarray(values, dtype=float)
        if len(values) < window:
            return episodes, values
        return episodes[window - 1:], np.convolve(values, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig.suptitle(title, fontsize=12)

    rewards = [row["total_reward"] for row in rows]
    ax = axes[0][0]
    ax.plot(episodes, rewards, color=color, alpha=0.25, linewidth=1, label="per episode")
    ax.plot(*moving_average(rewards), color=color, linewidth=2, label="50-episode average")
    validation = [(row["episode"], row["validation_reward"]) for row in rows if "validation_reward" in row]
    if validation:
        ax.plot(*zip(*validation), color="#eb6834", linewidth=2, marker="o", markersize=4,
                label="validation (greedy)")
    ax.set_title("Total reward per episode", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    loss_points = [(row["episode"], row["avg_loss"]) for row in rows if row["avg_loss"] != ""]
    ax = axes[0][1]
    if loss_points:
        ax.plot(*zip(*loss_points), color=color, linewidth=1.5)
    ax.set_title("Average loss (Huber)", loc="left", fontsize=10)

    ax = axes[1][0]
    ax.plot(episodes, [row["epsilon"] for row in rows], color=color, linewidth=2)
    ax.set_title("Epsilon at episode end", loc="left", fontsize=10)
    ax.set_xlabel("Episode")

    ax = axes[1][1]
    ax.plot(*moving_average([row["severe_failures"] for row in rows]), color=color, linewidth=2)
    ax.set_title("Severe failures per episode (50-episode average)", loc="left", fontsize=10)
    ax.set_xlabel("Episode")

    for ax in axes.flat:
        ax.grid(color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bdbcb6")
        ax.tick_params(colors="#52514e", labelsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()

    config = copy.deepcopy(APP_CONFIG)
    if args.learning_rate is not None:
        config["dqn"]["learning_rate"] = args.learning_rate
    if args.gamma is not None:
        config["dqn"]["gamma"] = args.gamma
    dqn_cfg = config["dqn"]
    episodes = args.episodes or dqn_cfg["episodes"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # The network is tiny: one CPU thread is faster than paying the cost of splitting work.
    torch.set_num_threads(1)

    run_name = args.run_name or f"{args.epsilon_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "config": config}, f, indent=2)

    # The run keeps its own copy of the forecaster, so evaluating or promoting it later uses
    # exactly the LSTM the agent learned with.
    lstm_path = None
    if not args.no_lstm:
        if not os.path.exists(args.lstm):
            raise SystemExit(f"{args.lstm} not found: run train_lstm.py first, or pass --no-lstm")
        lstm_path = os.path.join(run_dir, "lstm.pt")
        shutil.copyfile(args.lstm, lstm_path)
    predictor = Predictor.from_checkpoint(config, lstm_path)

    env = MockKubernetesEnv(config, seed=args.seed)
    agent = DQNAgent(STATE_SIZE, NUM_ACTIONS, dqn_cfg, seed=args.seed)
    epsilon_schedule = EpsilonSchedule(args.epsilon_mode, dqn_cfg)

    checkpoint_path = os.path.join(run_dir, "dqn.pt")            # best on validation
    last_checkpoint_path = os.path.join(run_dir, "dqn_last.pt")  # end of training
    replay_path = os.path.join(run_dir, "replay_buffer.pkl")
    checkpoint_extra = {"run_name": run_name, "epsilon_mode": args.epsilon_mode, "dqn": dqn_cfg}

    print(f"Training run '{run_name}': {episodes} episodes, epsilon mode '{args.epsilon_mode}', "
          f"lr={dqn_cfg['learning_rate']}, gamma={dqn_cfg['gamma']}, "
          f"forecaster={'LSTM' if lstm_path else 'naive'}")
    start_time = time.time()
    rows = []
    total_steps = 0

    validation_predictor = Predictor.from_checkpoint(config, lstm_path)
    best_validation = float("-inf")

    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for episode in range(1, episodes + 1):
            row = {"episode": episode,
                   **run_training_episode(env, predictor, agent, epsilon_schedule, total_steps)}
            total_steps = row["total_steps"]

            # Model selection: DQN training is noisy, and the last network is not always the
            # best one. Every few episodes the greedy policy is tested on fixed validation
            # episodes, and the best network so far is what dqn.pt keeps.
            if episode % dqn_cfg["validate_every_episodes"] == 0:
                row["validation_reward"] = round(validate(config, agent, validation_predictor), 2)
                if row["validation_reward"] > best_validation:
                    best_validation = row["validation_reward"]
                    agent.save(checkpoint_path, extra={**checkpoint_extra, "episode": episode,
                                                       "validation_reward": best_validation})

            rows.append(row)
            writer.writerow(row)

            if episode % 10 == 0:
                metrics_file.flush()
                recent = rows[-10:]
                print(f"Episode {episode:5d} | avg reward {np.mean([r['total_reward'] for r in recent]):9.1f} "
                      f"| failures {np.mean([r['severe_failures'] for r in recent]):5.1f} "
                      f"| replicas {np.mean([r['avg_replicas'] for r in recent]):5.1f} "
                      f"| epsilon {row['epsilon']:.3f} | best validation {best_validation:8.1f} "
                      f"| {time.time() - start_time:6.0f}s")

            if episode % dqn_cfg["save_every_episodes"] == 0:
                agent.save(last_checkpoint_path, extra={**checkpoint_extra, "episode": episode})
                if not args.no_replay_save:
                    agent.replay.save(replay_path)

    agent.save(last_checkpoint_path, extra={**checkpoint_extra, "episode": episodes})
    if best_validation == float("-inf"):   # run too short to validate: keep the last network
        agent.save(checkpoint_path, extra={**checkpoint_extra, "episode": episodes})
    if not args.no_replay_save:
        agent.replay.save(replay_path)
    plot_learning_curve(
        rows, os.path.join(run_dir, "learning_curve.png"),
        f"{run_name}  |  epsilon: {args.epsilon_mode}  |  lr={dqn_cfg['learning_rate']}  gamma={dqn_cfg['gamma']}")

    print(f"Done in {time.time() - start_time:.0f}s. Results in {run_dir}")


if __name__ == "__main__":
    main()
