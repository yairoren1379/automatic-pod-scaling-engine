"""The actions the agent can choose.

The numbers are the indexes of the DQN's output neurons, so their order is part of every saved
checkpoint. That is why they are fixed here in code and not editable in ZooKeeper.
"""

SCALE_UP = 0
SCALE_DOWN = 1
HOLD = 2

ACTION_NAMES = ["ScaleUp", "ScaleDown", "Hold"]
NUM_ACTIONS = len(ACTION_NAMES)


def action_id(name: str) -> int:
    """"ScaleUp" -> 0. Raises ValueError for an unknown name."""
    return ACTION_NAMES.index(name)
