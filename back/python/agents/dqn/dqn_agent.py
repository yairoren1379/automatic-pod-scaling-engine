"""Deep Q-Network agent: chooses scaling actions and learns from experience."""
import random

import torch  
from torch import nn

from agents.dqn.network import QNetwork
from agents.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """Learns Q(state, action): the total future reward expected after taking an action.

    Two networks with the same shape:
    - online network: chooses actions and is trained every step.
    - target network: a frozen copy used only to compute the learning target, synced from the
      online network every `target_sync_steps`. Without it the target would move with every
      gradient step (the network chasing its own tail), which often makes DQN diverge.
    """

    def __init__(self, state_size: int, num_actions: int, dqn_cfg: dict, seed: int = None):
        self.num_actions = num_actions
        self.gamma = dqn_cfg["gamma"]
        self.double_dqn = dqn_cfg.get("double_dqn", False)
        self.batch_size = dqn_cfg["batch_size"]
        self.min_replay_size = dqn_cfg["min_replay_size"]
        self.target_sync_steps = dqn_cfg["target_sync_steps"]
        self.grad_clip_norm = dqn_cfg["grad_clip_norm"]
        self.rng = random.Random(seed)

        hidden_size = dqn_cfg["hidden_size"]
        self.online_network = QNetwork(state_size, num_actions, hidden_size)
        self.target_network = QNetwork(state_size, num_actions, hidden_size)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=dqn_cfg["learning_rate"])
        # Huber loss: squared for small errors, linear for large ones. A -100 failure then
        # produces a firm but bounded gradient instead of a huge one (as MSE would).
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(dqn_cfg["replay_capacity"], seed=seed)
        self.learn_steps = 0

    # ---------------------------------------------------------------------------------------
    # Acting
    # ---------------------------------------------------------------------------------------

    def q_values(self, state) -> list:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.online_network(state_tensor).squeeze(0).tolist()

    def select_action(self, state, epsilon: float) -> int:
        """Epsilon-greedy: a random action with probability epsilon, otherwise the best one."""
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.num_actions)
        q_values = self.q_values(state)
        return max(range(self.num_actions), key=lambda action: q_values[action])

    # ---------------------------------------------------------------------------------------
    # Learning
    # ---------------------------------------------------------------------------------------

    def remember(self, state, action: int, reward: float, next_state, done: bool):
        self.replay.add(state, action, reward, next_state, done)

    def learn(self):
        """One gradient step on a random batch. Returns the loss, or None while warming up."""
        if len(self.replay) < self.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)

        # Q(s, a) for the action that was actually taken in each transition.
        predicted_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD target: r + gamma * max_a' Q_target(s', a'). A terminal state has no future.
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: the online network picks the best next action, the target network
                # rates it. Using one network for both always picks the action whose value is
                # most overestimated, and that error compounds through the TD targets.
                best_next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
                best_next_q = self.target_network(next_states).gather(1, best_next_actions).squeeze(1)
            else:
                best_next_q = self.target_network(next_states).max(dim=1).values
            target_q = rewards + self.gamma * best_next_q * (1 - dones)

        # Loss on the gap between prediction and target, then backpropagation.
        loss = self.loss_fn(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync_steps == 0:
            self.sync_target_network()

        return loss.item()

    def sync_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def set_learning_rate(self, learning_rate: float):
        """Changes the step size without losing the optimizer's state (used for fine-tuning)."""
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    # ---------------------------------------------------------------------------------------
    # Checkpoints
    # ---------------------------------------------------------------------------------------

    def save(self, path: str, extra: dict = None):
        torch.save({
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_steps": self.learn_steps,
            "extra": extra or {},
        }, path)

    def load(self, path: str) -> dict:
        """Restores the networks and optimizer. Returns the `extra` dict saved with them."""
        checkpoint = torch.load(path, weights_only=False)
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learn_steps = checkpoint["learn_steps"]
        return checkpoint["extra"]
