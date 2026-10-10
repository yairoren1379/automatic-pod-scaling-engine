"""Experience replay: a memory of past transitions that the agent learns from in random batches."""
import pickle
import random

import numpy as np


class ReplayBuffer:
    """A fixed-size circular buffer of (state, action, reward, next_state, done) transitions.

    Why replay instead of learning from each step as it happens: consecutive steps are almost
    identical, and training a network on such correlated data makes it unstable and forgetful.
    Sampling random batches from a large memory breaks the correlation and reuses every
    experience many times.
    """

    def __init__(self, capacity: int, seed: int = None):
        self.capacity = capacity
        self.transitions = []
        self.next_index = 0   # where the next transition is written once the buffer is full
        self.rng = random.Random(seed)

    def add(self, state, action: int, reward: float, next_state, done: bool):
        transition = (state, action, reward, next_state, done)
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            # Full: overwrite the oldest transition.
            self.transitions[self.next_index] = transition
        self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.transitions, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.transitions)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"transitions": self.transitions, "next_index": self.next_index}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.transitions = data["transitions"][-self.capacity:]
        self.next_index = data["next_index"] % self.capacity
