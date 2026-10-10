#finds the best action using epsilon-greedy bandit algorithm
import random
from typing import List
from config_loader import APP_CONFIG


class EpsilonGreedyBandit:
    def __init__(
        self,
        num_states: int,
        arms_count: int,
        epsilon: float = APP_CONFIG["rl_hyperparameters"]["epsilon"]
        ):
        self.num_states = num_states
        self.arms = arms_count
        self.epsilon = epsilon

        # [state][action] -> Q value
        self.q_values: List[List[float]] = [[APP_CONFIG["rl_hyperparameters"]["q_value_init"]] * arms_count for _ in range(num_states)]
        self.action_counts: List[List[int]] = [[APP_CONFIG["logic_constants"]["action_count_init"]] * arms_count for _ in range(num_states)]

    # returns the action with the highest Q value if the probability is higher than epsilon, else a random action
    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(APP_CONFIG["logic_constants"]["random_range_start"], APP_CONFIG["logic_constants"]["offset_to_last_index"])
        
        state_q_values = self.q_values[state]
        max_q = max(state_q_values)
        
        candidates = []
        for i in range(len(state_q_values)):
            if state_q_values[i] == max_q:
                candidates.append(i)
                
        # in case of 2 or more actions having the same Q value, randomly select one of them
        return random.choice(candidates)

    # gets the action, the reward and the state and updates the Q value of the action
    def updateAction(self, state: int, action: int, reward: float):
        self.action_counts[state][action] += 1
        action_counts = self.action_counts[state][action]

        old_q = self.q_values[state][action]
        
        learning_rate = max(APP_CONFIG["logic_constants"]["min_learning_rate"], APP_CONFIG["logic_constants"]["update_factor_numerator"] / action_counts)
        
        new_q = old_q + learning_rate * (reward - old_q)
        self.q_values[state][action] = new_q

    def __repr__(self):
        return f"EpsilonGreedyBandit(states={self.num_states}, arms={self.arms})"