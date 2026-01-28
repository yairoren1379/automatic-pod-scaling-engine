#learning the best actions using q-learning algorithm
from typing import List, Optional
import random
from agents.config import RLConfig


class QLearningAgent:

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = RLConfig.ALPHA,   # מקדם למידה how fast the agent learns
        gamma: float = RLConfig.GAMMA,  # פקטור הנחה how much future rewards are valued
        epsilon: float = RLConfig.EPSILON, # הסתברות לחקירה how often to explore
    ):
        self.num_states = num_states
        self.num_actions = num_actions # [scale up, scale down, nothing, restart]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # every cell represents the Q value for a (state, action) pair
        self.q_table = []
        for state_index in range(num_states):
            row = []
            for action_index in range(num_actions):
                row.append(RLConfig.Q_VALUE_INIT)
            self.q_table.append(row)
            
    def select_action(
        self,
        state: int,
        allowed_actions: Optional[List[int]] = None,
    ) -> int:

        if allowed_actions is None:
            allowed_actions = list(range(self.num_actions))

        # חקירה
        if random.random() < self.epsilon:
            return random.choice(allowed_actions)

        # choose the action with the highest Q value
        q_values = self.q_table[state]
        max_q = float('-inf')
        for a in allowed_actions:
            max_q = max(max_q, q_values[a])

        # in case of 2 or more actions having the same Q value, randomly select one of them
        candidates = []
        for a in allowed_actions:
            if q_values[a] == max_q:
                candidates.append(a)
                
        return random.choice(candidates)

    # learns from the action taken and the reward received
    # and updates the Q value accordingly
    def updateAction(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ):
        
        old_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state])
            # the reward plus the future discounted reward
            target = reward + self.gamma * max_next_q

        # change the Q value a little bit towards the target
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[state][action] = new_q

    def __repr__(self) -> str:
        return f"QLearningAgent(alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon})"
