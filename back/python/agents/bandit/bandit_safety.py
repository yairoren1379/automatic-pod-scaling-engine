# filters actions based on their safety and catastrophic failure rates
from typing import List
from config_loader import APP_CONFIG
from agents.bandit.bandit import EpsilonGreedyBandit


class SafetyBandit(EpsilonGreedyBandit):
    def __init__(
        self,
        num_states: int,
        arms_count: int,
        epsilon: float = APP_CONFIG["rl_hyperparameters"]["epsilon"],
        catastrophic_penalty: float = APP_CONFIG["rl_hyperparameters"]["catastrophic_penalty"],
        safe_reward: float = APP_CONFIG["rewards"]["safe_reward"],
    ):
        super().__init__(num_states=num_states, arms_count=arms_count, epsilon=epsilon)
        self.catastrophic_penalty = catastrophic_penalty
        self.safe_reward = safe_reward

        self.failure_counts: List[List[int]] = [[APP_CONFIG["logic_constants"]["failure_count_init"]] * self.arms for _ in range(num_states)]

    # gives bad reward if the outcome is a catastrophic failure
    # else gives safe reward
    def update_from_outcome(self, state: int, action: int, is_catastrophic_failure: bool):
        if is_catastrophic_failure:
            reward = self.catastrophic_penalty
            self.failure_counts[state][action] += APP_CONFIG["logic_constants"]["failure_count_increment"]
        else:
            reward = self.safe_reward

        self.updateAction(state, action, reward)

    def get_safe_actions(self,state: int, max_failure_rate: float, min_tries: int = APP_CONFIG["logic_constants"]["min_tries_default"]) -> List[int]:
        safe = []
        for i in range(self.arms):
            total_count = self.action_counts[state][i]
            if total_count < min_tries:
                safe.append(i)
                continue
            
            failures_count = self.failure_counts[state][i]
            failure_rate = failures_count / total_count
            
            if failure_rate <= max_failure_rate:
                safe.append(i)

        return safe
    

