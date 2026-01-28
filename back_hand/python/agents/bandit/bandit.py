#finds the best action using epsilon-greedy bandit algorithm
import random
from typing import List

from agents.config import RLConfig

ACTION_COUNT_INIT = 0
RANDOM_RANGE_START = 0
OFFSET_TO_LAST_INDEX = 1 # used to not go out of range
UPDATE_FACTOR_NUMERATOR = 1.0

class EpsilonGreedyBandit:
    def __init__(self, arms_count: int, epsilon: float = RLConfig.EPSILON):
        self.arms = arms_count
        self.epsilon = epsilon

        # [scale up, scale down, nothing, restart]
        self.q_values: List[float] = [RLConfig.Q_VALUE_INIT] * arms_count
        self.action_counts: List[int] = [ACTION_COUNT_INIT] * arms_count

    # returns the action with the highest Q value if the probability is higher than epsilon, else a random action
    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(RANDOM_RANGE_START, self.arms - OFFSET_TO_LAST_INDEX)
        
        max_q = max(self.q_values)
        candidates = []
        for i in range(len(self.q_values)):
            if self.q_values[i] == max_q:
                candidates.append(i)
                
        # in case of 2 or more actions having the same Q value, randomly select one of them
        return random.choice(candidates)

    # gets the action and the reward and updates the Q value of the action
    def updateAction(self, action: int, reward: float):
        self.action_counts[action] += 1
        action_counts = self.action_counts[action]

        old_q = self.q_values[action]
        new_q = old_q + (UPDATE_FACTOR_NUMERATOR / action_counts) * (reward - old_q)
        self.q_values[action] = new_q

    def __repr__(self):
        return f"EpsilonGreedyBandit(q_values={self.q_values}, counts={self.action_counts})"