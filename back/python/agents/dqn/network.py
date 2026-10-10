"""The Q-network: maps a state vector to one Q value per action."""
import torch
from torch import nn


class QNetwork(nn.Module):
    """A small fully connected network: state -> hidden -> hidden -> Q(state, action) for each action.

    The old agent kept a table with one row per (CPU bucket, RAM bucket, replicas) combination.
    A network instead generalizes: states it never visited still get sensible Q values because
    they are close to states it did visit. This is what lets the state hold continuous metrics.
    """

    def __init__(self, state_size: int, num_actions: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
