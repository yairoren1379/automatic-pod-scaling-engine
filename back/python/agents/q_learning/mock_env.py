#train and chack q-learning agent in a mock kubernetes environment
import random
from typing import Tuple
from config_loader import APP_CONFIG

class MockKubernetesEnv:
    def __init__(self):
        self.min_pods = APP_CONFIG["system_limits"]["min_pods"]
        self.max_pods = APP_CONFIG["system_limits"]["max_pods"]
        self.num_buckets = APP_CONFIG["metrics_config"]["num_buckets"]
        
        self.cpu_bucket = self.num_buckets // 2
        self.ram_bucket = self.num_buckets // 2
        self.replicas = APP_CONFIG["logic_constants"]["initial_replicas"]
        self.step_count = APP_CONFIG["logic_constants"]["initial_step_count"]
        self.max_steps = APP_CONFIG["rl_hyperparameters"]["max_steps"]

    def _encode_state(self) -> int:
        # מחשבים רק את טווח הפודים החוקי (למשל מ-1 עד 15 זה 15 מצבים, ולא 16)
        valid_pod_states = self.max_pods - self.min_pods + 1
        
        # מתקנים את האינדקס כדי שיתחיל מ-0 במקום מ-min_pods
        pod_index = self.replicas - self.min_pods
        
        return (self.cpu_bucket * self.num_buckets + self.ram_bucket) * valid_pod_states + pod_index

    # resets the environment
    # returns the initial state
    def reset(self) -> int:
        self.cpu_bucket = random.choice(range(self.num_buckets))
        self.ram_bucket = random.choice(range(self.num_buckets))
        self.replicas = random.randint(self.min_pods, self.max_pods)
        self.step_count = APP_CONFIG["logic_constants"]["initial_step_count"]
        return self._encode_state()
    
    def is_failure(self, action: int) -> bool:
        if action == APP_CONFIG["actions"]["scale_down"] and self.replicas <= self.min_pods:
            return True
        if action == APP_CONFIG["actions"]["scale_up"] and self.replicas >= self.max_pods:
            return True
        # Failure condition to match real world RAM crash
        if (self.cpu_bucket >= self.num_buckets - 2 or self.ram_bucket >= self.num_buckets - 2) and self.replicas <= 2:
            if action != APP_CONFIG["actions"]["scale_up"]:
                return True
        return False

    def _apply_action_effects(self, replica_delta: int, load_delta: int):
        self.replicas = max(self.min_pods, min(self.max_pods, self.replicas + replica_delta))
        
        max_bucket = self.num_buckets - 1
        min_bucket = APP_CONFIG["logic_constants"]["min_level"]
        self.cpu_bucket = max(min_bucket, min(max_bucket, self.cpu_bucket + load_delta))
        self.ram_bucket = max(min_bucket, min(max_bucket, self.ram_bucket + load_delta))
        
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        self.step_count += APP_CONFIG["logic_constants"]["step_size"]

        # -1 - less cpu, 0 - same, +1 - more cpu
        noise_cpu = random.choice([-APP_CONFIG["logic_constants"]["step_size"], 0, APP_CONFIG["logic_constants"]["step_size"]])
        noise_ram = random.choice([-APP_CONFIG["logic_constants"]["step_size"], 0, APP_CONFIG["logic_constants"]["step_size"]])
        
        self.cpu_bucket = min(self.num_buckets - 1, max(APP_CONFIG["logic_constants"]["min_level"], self.cpu_bucket + noise_cpu))
        self.ram_bucket = min(self.num_buckets - 1, max(APP_CONFIG["logic_constants"]["min_level"], self.ram_bucket + noise_ram))

        step_size = APP_CONFIG["logic_constants"]["step_size"]
        load_effect = step_size

        if action == APP_CONFIG["actions"]["scale_up"]:
            self._apply_action_effects(step_size, -load_effect)

        elif action == APP_CONFIG["actions"]["scale_down"]:
            self._apply_action_effects(-step_size, load_effect)

        elif action == APP_CONFIG["actions"]["restart"]:
            self._apply_action_effects(0, load_effect)

        elif action == APP_CONFIG["actions"]["no_action"]:
            pass
        
        #starts with neutral reward
        reward = APP_CONFIG["logic_constants"]["initial_reward"]

        #ideal state
        if self.cpu_bucket == APP_CONFIG["logic_constants"]["ideal_cpu_level"] and self.replicas == APP_CONFIG["logic_constants"]["ideal_replicas"]:
            reward += APP_CONFIG["rewards"]["mock_ideal"]

        #high load
        if self.cpu_bucket >= self.num_buckets - 2:
            reward += APP_CONFIG["rewards"]["mock_high_load"]
        if self.ram_bucket >= self.num_buckets - 3:
            reward += APP_CONFIG["rewards"]["mock_high_load"] * 2

        # waste of resources
        if self.cpu_bucket <= APP_CONFIG["logic_constants"]["min_level"] and self.replicas >= self.max_pods - 2:
            reward += APP_CONFIG["rewards"]["mock_waste"]

        # reset penalty
        if action == APP_CONFIG["actions"]["restart"]:
            reward += APP_CONFIG["rewards"]["mock_restart_penalty"]

        done = self.step_count >= self.max_steps

        next_state = self._encode_state()
        info = {
            "cpu_bucket": self.cpu_bucket,
            "ram_bucket": self.ram_bucket,
            "replicas": self.replicas,
        }
        return next_state, reward, done, info