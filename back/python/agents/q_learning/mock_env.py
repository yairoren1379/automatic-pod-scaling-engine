#train and chack q-learning agent in a mock kubernetes environment
import random
from typing import Tuple
from config_loader import APP_CONFIG

class MockKubernetesEnv:
    def __init__(self):
        self.cpu_level = APP_CONFIG["logic_constants"]["initial_cpu_level"]
        self.replicas = APP_CONFIG["logic_constants"]["initial_replicas"]
        self.step_count = APP_CONFIG["logic_constants"]["initial_step_count"]
        self.max_steps = APP_CONFIG["rl_hyperparameters"]["max_steps"]

    # convert state to a single integer in base 3 from 2 dim to 1 dim
    def _encode_state(self) -> int:
        return self.cpu_level * APP_CONFIG["levels"]["count"] + self.replicas

    # resets the environment
    # returns the initial state
    def reset(self) -> int:
        self.cpu_level = random.choice(range(APP_CONFIG["levels"]["count"]))
        self.replicas = APP_CONFIG["logic_constants"]["initial_replicas"]
        self.step_count = APP_CONFIG["logic_constants"]["initial_step_count"]
        return self._encode_state()
    
    def is_failure(self, action: int) -> bool:
        if action == APP_CONFIG["actions"]["scale_down"] and self.cpu_level == APP_CONFIG["levels"]["count"] - 1:
            return True
        if action == APP_CONFIG["actions"]["scale_down"] and self.replicas == APP_CONFIG["logic_constants"]["min_level"]:
            return True
        return False

    def step(self, action: int) -> Tuple[int, float, bool, dict]:

        self.step_count += APP_CONFIG["logic_constants"]["step_size"]

        # -1 - less cpu, 0 - same, +1 - more cpu
        noise = random.choice([-APP_CONFIG["logic_constants"]["step_size"], 0, APP_CONFIG["logic_constants"]["step_size"]])
        self.cpu_level = min(APP_CONFIG["levels"]["count"] - 1, max(APP_CONFIG["logic_constants"]["min_level"], self.cpu_level + noise))

        if action == APP_CONFIG["actions"]["scale_up"]:
            self.replicas = min(APP_CONFIG["levels"]["count"] - 1, self.replicas + APP_CONFIG["logic_constants"]["step_size"])
            if self.cpu_level > APP_CONFIG["logic_constants"]["min_level"]:
                self.cpu_level -= APP_CONFIG["logic_constants"]["step_size"]

        elif action == APP_CONFIG["actions"]["scale_down"]:
            self.replicas = max(APP_CONFIG["logic_constants"]["min_level"], self.replicas - APP_CONFIG["logic_constants"]["step_size"])
            if self.cpu_level < APP_CONFIG["levels"]["count"] - 1:
                self.cpu_level += APP_CONFIG["logic_constants"]["step_size"]

        elif action == APP_CONFIG["actions"]["restart"]:
            if self.cpu_level < APP_CONFIG["levels"]["count"] - 1:
                self.cpu_level += APP_CONFIG["logic_constants"]["step_size"]
                
        elif action == APP_CONFIG["actions"]["no_action"]:
            pass
        
        #starts with neutral reward
        reward = APP_CONFIG["logic_constants"]["initial_reward"]

        #ideal state
        if self.cpu_level == APP_CONFIG["logic_constants"]["ideal_cpu_level"] and self.replicas == APP_CONFIG["logic_constants"]["ideal_replicas"]:
            reward += APP_CONFIG["rewards"]["mock_ideal"]

        #high load
        if self.cpu_level == APP_CONFIG["levels"]["count"] - 1:
            reward += APP_CONFIG["rewards"]["mock_high_load"]

        # waste of resources
        if self.cpu_level == APP_CONFIG["logic_constants"]["min_level"] and self.replicas == APP_CONFIG["levels"]["count"] - 1:
            reward += APP_CONFIG["rewards"]["mock_waste"]

        # reset penalty
        if action == APP_CONFIG["actions"]["restart"]:
            reward += APP_CONFIG["rewards"]["mock_restart_penalty"]

        done = self.step_count >= self.max_steps

        next_state = self._encode_state()
        info = {
            "cpu_level": self.cpu_level,
            "replicas": self.replicas,
        }
        return next_state, reward, done, info