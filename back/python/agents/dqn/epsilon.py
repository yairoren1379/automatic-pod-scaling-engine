"""Epsilon schedules: how often the agent explores a random action instead of its best one."""


class EpsilonSchedule:
    """Two modes, compared in the experiments:

    - "decay":    eps = max(eps_min, eps_0 / (k*t + 1)). Explores a lot at first, then gradually
                  trusts what it learned. t is the number of environment steps so far.
    - "constant": eps never changes, so the agent keeps exploring at the same rate.
    """

    def __init__(self, mode: str, dqn_cfg: dict):
        if mode not in ("decay", "constant"):
            raise ValueError(f"Unknown epsilon mode: {mode}")
        self.mode = mode
        self.epsilon_start = dqn_cfg["epsilon_start"]
        self.epsilon_min = dqn_cfg["epsilon_min"]
        self.k = dqn_cfg["epsilon_decay_k"]
        self.epsilon_constant = dqn_cfg["epsilon_constant"]

    def value(self, step: int) -> float:
        if self.mode == "constant":
            return self.epsilon_constant
        return max(self.epsilon_min, self.epsilon_start / (self.k * step + 1))
