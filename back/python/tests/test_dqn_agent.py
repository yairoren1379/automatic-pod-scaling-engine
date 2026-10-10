import numpy as np
import torch

from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.epsilon import EpsilonSchedule
from agents.dqn.replay_buffer import ReplayBuffer


def small_agent(config):
    cfg = dict(config["dqn"], min_replay_size=32, batch_size=32, hidden_size=16, learning_rate=0.01)
    return DQNAgent(state_size=3, num_actions=3, dqn_cfg=cfg, seed=0)


def test_no_learning_until_buffer_is_warm(config):
    agent = small_agent(config)
    agent.remember(np.zeros(3), 0, 1.0, np.zeros(3), False)
    assert agent.learn() is None


def test_agent_learns_which_action_pays(config):
    """One state, action 1 always pays +1, the others -1: the greedy choice must become 1."""
    torch.manual_seed(0)
    agent = small_agent(config)
    agent.gamma = 0.0
    state = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    for _ in range(300):
        action = agent.rng.randrange(3)
        agent.remember(state, action, 1.0 if action == 1 else -1.0, state, False)
        agent.learn()
    assert agent.select_action(state, epsilon=0.0) == 1


def test_epsilon_decay_formula(config):
    schedule = EpsilonSchedule("decay", config["dqn"])
    k = config["dqn"]["epsilon_decay_k"]
    assert schedule.value(0) == 1.0
    assert schedule.value(1000) == max(config["dqn"]["epsilon_min"], 1.0 / (k * 1000 + 1))
    assert schedule.value(10**7) == config["dqn"]["epsilon_min"]
    assert EpsilonSchedule("constant", config["dqn"]).value(10**7) == config["dqn"]["epsilon_constant"]


def test_replay_buffer_overwrites_oldest(tmp_path):
    buffer = ReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        buffer.add([i], i, float(i), [i + 1], False)
    assert len(buffer) == 3
    assert sorted(t[1] for t in buffer.transitions) == [2, 3, 4]

    buffer.save(tmp_path / "replay.pkl")
    restored = ReplayBuffer(capacity=3)
    restored.load(tmp_path / "replay.pkl")
    assert restored.transitions == buffer.transitions


def test_checkpoint_round_trip(config, tmp_path):
    agent = small_agent(config)
    state = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    agent.save(tmp_path / "dqn.pt", extra={"note": "test"})
    other = small_agent(config)
    assert other.load(tmp_path / "dqn.pt") == {"note": "test"}
    assert other.q_values(state) == agent.q_values(state)
