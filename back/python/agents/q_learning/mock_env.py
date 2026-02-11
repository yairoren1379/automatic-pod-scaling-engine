#train and chack q-learning agent in a mock kubernetes environment
import random
from typing import Tuple

OFFSET_TO_LAST_INDEX = 1 # used to not go out of range when indexing

# Actions
ACTION_SCALE_UP = 0
ACTION_SCALE_DOWN = 1
ACTION_NOTHING = 2
ACTION_RESTART = 3

# Environment Configuration
MAX_STEPS = 100
CPU_LEVELS = 3       # 0, 1, 2
REPLICA_LEVELS = 3   # 0, 1, 2

# Steps
MIN_LEVEL = 0
STEP_SIZE = 1

# Ideal State
IDEAL_CPU_LEVEL = 1
IDEAL_REPLICAS = 1

# Initial State
INITIAL_CPU_LEVEL = 1
INITIAL_REPLICAS = 1
INITIAL_STEP_COUNT = 0
INITIAL_REWARD = 0.0

# Rewards
REWARD_IDEAL = 5.0
REWARD_HIGH_LOAD = -3.0
REWARD_WASTE = -2.0
REWARD_RESTART_PENALTY = -1.0

class MockKubernetesEnv:
    def __init__(self):
        self.cpu_level = INITIAL_CPU_LEVEL
        self.replicas = INITIAL_REPLICAS
        self.step_count = INITIAL_STEP_COUNT
        self.max_steps = MAX_STEPS

    # convert state to a single integer in base 3 from 2 dim to 1 dim
    def _encode_state(self) -> int:
        return self.cpu_level * REPLICA_LEVELS + self.replicas

    # resets the environment
    # returns the initial state
    def reset(self) -> int:
        self.cpu_level = random.choice(range(CPU_LEVELS))
        self.replicas = INITIAL_REPLICAS
        self.step_count = INITIAL_STEP_COUNT
        return self._encode_state()
    
    def is_failure(self, action: int) -> bool:
        if action == ACTION_SCALE_DOWN and self.cpu_level == CPU_LEVELS - OFFSET_TO_LAST_INDEX:
            return True
        if action == ACTION_SCALE_DOWN and self.replicas == MIN_LEVEL:
            return True
        return False

    def step(self, action: int) -> Tuple[int, float, bool, dict]:

        self.step_count += STEP_SIZE

        # -1 - less cpu, 0 - same, +1 - more cpu
        noise = random.choice([-STEP_SIZE, 0, STEP_SIZE])
        self.cpu_level = min(CPU_LEVELS - OFFSET_TO_LAST_INDEX, max(MIN_LEVEL, self.cpu_level + noise))

        if action == ACTION_SCALE_UP:
            self.replicas = min(REPLICA_LEVELS - OFFSET_TO_LAST_INDEX, self.replicas + STEP_SIZE)
            if self.cpu_level > MIN_LEVEL:
                self.cpu_level -= STEP_SIZE

        elif action == ACTION_SCALE_DOWN:
            self.replicas = max(MIN_LEVEL, self.replicas - STEP_SIZE)
            if self.cpu_level < CPU_LEVELS - OFFSET_TO_LAST_INDEX:
                self.cpu_level += STEP_SIZE

        elif action == ACTION_RESTART:
            if self.cpu_level < CPU_LEVELS - OFFSET_TO_LAST_INDEX:
                self.cpu_level += STEP_SIZE
                
        elif action == ACTION_NOTHING:
            pass
        
        #starts with neutral reward
        reward = INITIAL_REWARD

        #ideal state
        if self.cpu_level == IDEAL_CPU_LEVEL and self.replicas == IDEAL_REPLICAS:
            reward += REWARD_IDEAL

        #high load
        if self.cpu_level == CPU_LEVELS - OFFSET_TO_LAST_INDEX:
            reward += REWARD_HIGH_LOAD

        # waste of resources
        if self.cpu_level == MIN_LEVEL and self.replicas == REPLICA_LEVELS - OFFSET_TO_LAST_INDEX:
            reward += REWARD_WASTE

        # reset penalty
        if action == ACTION_RESTART:
            reward += REWARD_RESTART_PENALTY

        done = self.step_count >= self.max_steps

        next_state = self._encode_state()
        info = {
            "cpu_level": self.cpu_level,
            "replicas": self.replicas,
        }
        return next_state, reward, done, info