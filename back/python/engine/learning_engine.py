"""The learning engine behind the API: decides, learns online, records everything.

One control loop of the Go controller makes two calls:

    1. decide(mode, metrics)  The Predictor builds the state. The previous decision's outcome is
                              now known, so its transition (state, executed action, reward,
                              new state) is completed, stored and learned from. Then the DQN
                              proposes an action for the new state.
    2. report(...)            The controller says what it actually executed and whether a
                              guardrail blocked it. This is written to the audit log.

What the modes mean here:
    baseline  The threshold controller drives the cluster. Nothing is learned; the DQN's
              proposal is only recorded, for comparison.
    shadow    The threshold controller still drives the cluster, and the DQN learns from what it
              executes (off-policy: Q-learning can learn from another policy's actions).
    active    The DQN's action is executed through the guardrails. It keeps fine-tuning with a
              lower learning rate and a small epsilon.
"""
import os
import threading
import time
from collections import deque
from dataclasses import asdict

from agents.dqn.dqn_agent import DQNAgent
from core.actions import ACTION_NAMES, NUM_ACTIONS, action_id
from core.reward import compute_reward
from core.state import STATE_SIZE, ClusterMetrics
from engine.storage import AuditLog, MetricsCsv
from predictor.predictor import Predictor

MODES = ("baseline", "shadow", "active")
LEARNING_MODES = ("shadow", "active")


class LearningEngine:

    def __init__(self, config: dict, models_dir: str, data_dir: str):
        self.lock = threading.Lock()
        self.models_dir, self.data_dir = models_dir, data_dir
        self.online_checkpoint = os.path.join(data_dir, "online_dqn.pt")
        self.online_replay = os.path.join(data_dir, "online_replay.pkl")
        self.metrics_csv = MetricsCsv(os.path.join(data_dir, "metrics.csv"))
        self.audit_log = AuditLog(os.path.join(data_dir, "audit.jsonl"))

        self.config = config
        self.agent = DQNAgent(STATE_SIZE, NUM_ACTIONS, config["dqn"])
        self.model_source = self._load_agent()
        lstm_path = os.path.join(models_dir, "lstm.pt")
        self.predictor = Predictor.from_checkpoint(config, lstm_path if os.path.exists(lstm_path) else None)
        self.forecaster_source = "LSTM" if self.predictor.forecaster else "naive (no models/lstm.pt)"
        self.apply_config(config)

        history_points = config["online"]["history_points"]
        self.history = deque(self.metrics_csv.tail(history_points), maxlen=history_points)
        self.audit = deque(self.audit_log.tail(200), maxlen=200)
        self.learning_curve = deque(maxlen=history_points)   # online reward / loss / epsilon
        self.pending = None           # the last decision, waiting for its report and outcome
        self.decision_counter = 0
        self.online_steps = 0         # transitions learned from on the real cluster
        self.last_contact = None
        self.status = {"mode": config["controller"]["mode"], "decision": None, "metrics": None,
                       "smoothed": None, "forecast": None, "regime": 0, "q_values": None,
                       "reward": None, "epsilon": 0.0}

    # ---------------------------------------------------------------------------------------
    # Model loading and configuration
    # ---------------------------------------------------------------------------------------

    def _load_agent(self) -> str:
        """Resume online learning if there is a checkpoint, else start from the sim-trained model."""
        if os.path.exists(self.online_checkpoint):
            self.agent.load(self.online_checkpoint)
            if os.path.exists(self.online_replay):
                self.agent.replay.load(self.online_replay)
            return f"online checkpoint ({self.online_checkpoint})"
        pretrained = os.path.join(self.models_dir, "dqn.pt")
        if os.path.exists(pretrained):
            self.agent.load(pretrained)
            # A fresh, empty replay buffer: online learning starts from real transitions only.
            return f"pre-trained in simulation ({pretrained})"
        return "untrained (no models/dqn.pt - run promote_model.py)"

    def apply_config(self, config: dict):
        """Applies a new configuration on the fly (called when the ZooKeeper ZNode changes)."""
        with self.lock:
            self.config = config
            self.predictor.config = config
            predictor_cfg = config["predictor"]
            hysteresis = self.predictor.hysteresis
            hysteresis.high_enter = predictor_cfg["high_enter_cpu_percent"]
            hysteresis.high_exit = predictor_cfg["high_exit_cpu_percent"]
            hysteresis.low_enter = predictor_cfg["low_enter_cpu_percent"]
            hysteresis.low_exit = predictor_cfg["low_exit_cpu_percent"]

            online = config["online"]
            self.agent.gamma = config["dqn"]["gamma"]
            self.agent.batch_size = online["batch_size"]
            self.agent.min_replay_size = online["min_replay_size"]

    def _epsilon(self, mode: str) -> float:
        return self.config["online"]["active_epsilon"] if mode == "active" else 0.0

    def _learning_rate(self, mode: str) -> float:
        online = self.config["online"]
        return online["active_learning_rate"] if mode == "active" else online["shadow_learning_rate"]

    # ---------------------------------------------------------------------------------------
    # The control loop
    # ---------------------------------------------------------------------------------------

    def decide(self, mode: str, metrics: ClusterMetrics) -> dict:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        with self.lock:
            self.last_contact = time.time()
            output = self.predictor.observe(metrics)
            reward = self._complete_previous_transition(metrics, output.state)

            epsilon = self._epsilon(mode)
            action = self.agent.select_action(output.state, epsilon)
            q_values = self.agent.q_values(output.state)

            self.decision_counter += 1
            self.pending = {
                "decision_id": self.decision_counter, "mode": mode, "state": output.state,
                "metrics": metrics, "output": output, "dqn_action": action, "q_values": q_values,
                "reward": reward, "executed_action": None, "blocked": False,
            }
            self.status.update({
                "mode": mode, "metrics": asdict(metrics), "smoothed": asdict(output.smoothed),
                "forecast": output.forecast, "regime": output.regime, "q_values": q_values,
                "decision": ACTION_NAMES[action], "epsilon": epsilon,
                "reward": asdict(reward) if reward else None,
            })
            return {
                "decision_id": self.decision_counter,
                "action": ACTION_NAMES[action],
                "q_values": dict(zip(ACTION_NAMES, q_values)),
                "epsilon": epsilon,
                "forecast": output.forecast,
                "regime": output.regime,
            }

    def _complete_previous_transition(self, metrics: ClusterMetrics, state):
        """The outcome of the previous action is the current metrics: compute its reward, and
        learn from the transition in the learning modes."""
        previous = self.pending
        if previous is None or previous["executed_action"] is None:
            return None   # first call, or the controller never reported what it executed

        reward = compute_reward(previous["metrics"], metrics, previous["blocked"], self.config)
        loss = None
        if previous["mode"] in LEARNING_MODES:
            self.agent.set_learning_rate(self._learning_rate(previous["mode"]))
            self.agent.remember(previous["state"], previous["executed_action"], reward.total, state, False)
            losses = [self.agent.learn() for _ in range(self.config["online"]["updates_per_step"])]
            losses = [l for l in losses if l is not None]
            loss = sum(losses) / len(losses) if losses else None
            self.online_steps += 1
            if self.online_steps % self.config["online"]["save_every_steps"] == 0:
                self.save()

        self.learning_curve.append({
            "timestamp": time.time(), "mode": previous["mode"], "reward": reward.total, "loss": loss,
            "epsilon": self._epsilon(previous["mode"]), "online_steps": self.online_steps,
            "replay_size": len(self.agent.replay),
        })
        return reward

    def report(self, decision_id: int, executed_action: str, source: str, blocked: bool,
               block_reason: str, replicas_before: int, replicas_after: int) -> dict:
        """Records what the controller executed. Returns the events to stream to the dashboard."""
        with self.lock:
            pending = self.pending
            if pending is None or pending["decision_id"] != decision_id:
                raise LookupError(f"decision {decision_id} is not the latest decision")
            pending["executed_action"] = action_id(executed_action)
            pending["blocked"] = blocked

            now = time.time()
            metrics, output = pending["metrics"], pending["output"]
            reward = pending["reward"]
            audit_entry = {
                "timestamp": now, "decision_id": decision_id, "mode": pending["mode"],
                "dqn_action": ACTION_NAMES[pending["dqn_action"]],
                "q_values": dict(zip(ACTION_NAMES, pending["q_values"])),
                "executed_action": executed_action, "source": source,
                "blocked": blocked, "block_reason": block_reason,
                "replicas_before": replicas_before, "replicas_after": replicas_after,
                "cpu_percent": metrics.cpu_percent, "latency_ms": metrics.latency_ms,
                "qps": metrics.qps,
            }
            point = {
                "timestamp": now, "mode": pending["mode"], **asdict(metrics),
                "regime": output.regime,
                **{f"forecast_{name}": value for name, value in output.forecast.items()},
                "dqn_action": ACTION_NAMES[pending["dqn_action"]], "executed_action": executed_action,
                "source": source, "blocked": blocked,
                "reward": reward.total if reward else None,
            }
            self.audit.append(audit_entry)
            self.audit_log.append(audit_entry)
            self.history.append(point)
            self.metrics_csv.append(point)
            return {"point": point, "audit": audit_entry,
                    "learning": self.learning_curve[-1] if self.learning_curve else None}

    # ---------------------------------------------------------------------------------------
    # Persistence and read access for the dashboard
    # ---------------------------------------------------------------------------------------

    def save(self):
        """Checkpoint for recovery: the fine-tuned networks and the real-world replay buffer."""
        os.makedirs(self.data_dir, exist_ok=True)
        self.agent.save(self.online_checkpoint, extra={"online_steps": self.online_steps})
        self.agent.replay.save(self.online_replay)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                **self.status,
                "connected": self.last_contact is not None and time.time() - self.last_contact < 3 * max(
                    1, self.config["controller"]["loop_delay_seconds"]),
                "last_contact": self.last_contact,
                "online_steps": self.online_steps,
                "replay_size": len(self.agent.replay),
                "learn_steps": self.agent.learn_steps,
                "model_source": self.model_source,
                "forecaster": self.forecaster_source,
            }

    def recent_history(self, limit: int) -> list:
        with self.lock:
            return list(self.history)[-limit:]

    def recent_audit(self, limit: int) -> list:
        with self.lock:
            return list(self.audit)[-limit:]

    def online_learning_curve(self) -> list:
        with self.lock:
            return list(self.learning_curve)
