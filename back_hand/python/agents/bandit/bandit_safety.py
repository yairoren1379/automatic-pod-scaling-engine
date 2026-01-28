# filters actions based on their safety and catastrophic failure rates
from typing import List

from agents.config import RLConfig
from agents.bandit.bandit import EpsilonGreedyBandit

FAILURE_COUNT_INIT = 0
FAILURE_COUNT_INCREMENT = 1
MIN_TRIES_DEFAULT = 10

class SafetyBandit(EpsilonGreedyBandit):
    def __init__(
        self,
        arms_count: int,
        epsilon: float = RLConfig.EPSILON,
        catastrophic_penalty: float = RLConfig.CATASTROPHIC_PENALTY,
        safe_reward: float = RLConfig.SAFE_REWARD,
    ):
        super().__init__(arms_count=arms_count, epsilon=epsilon)
        self.catastrophic_penalty = catastrophic_penalty
        self.safe_reward = safe_reward

        self.failure_counts: List[int] = [FAILURE_COUNT_INIT] * self.arms

    # gives bad reward if the outcome is a catastrophic failure
    # else gives safe reward
    def update_from_outcome(self, action: int, is_catastrophic_failure: bool):
        if is_catastrophic_failure:
            reward = self.catastrophic_penalty
            self.failure_counts[action] += FAILURE_COUNT_INCREMENT
        else:
            reward = self.safe_reward

        # uses the father class update method
        self.updateAction(action, reward)

    def get_safe_actions(self, max_failure_rate: float, min_tries: int = MIN_TRIES_DEFAULT) -> List[int]:
        safe = []
        for i in range(self.arms):
            if self.action_counts[i] < min_tries:
                continue

            failures_count = self.failure_counts[i]
            total_count = self.action_counts[i]
            failure_rate = failures_count / total_count

            if failure_rate <= max_failure_rate:
                safe.append(i)

        return safe