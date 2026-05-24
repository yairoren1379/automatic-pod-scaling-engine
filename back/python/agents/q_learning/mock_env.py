#train and chack q-learning agent in a mock kubernetes environment
import random
from typing import Tuple
from config_loader import APP_CONFIG

def calculate_reward(cpu_bucket: int, ram_bucket: int, replicas: int, action: int, last_action: int = None, done: bool = False) -> float:
    
    ideal_cpu = APP_CONFIG["logic_constants"]["ideal_cpu_level"]
    ideal_ram = APP_CONFIG["logic_constants"]["ideal_ram_level"]
    ideal_replicas = APP_CONFIG["logic_constants"]["ideal_replicas"]
    high_threshold = APP_CONFIG["logic_constants"].get("high_load_threshold", 24)
    waste_threshold = APP_CONFIG["logic_constants"].get("low_load_threshold", 7)
    
    min_pods = APP_CONFIG["system_limits"].get("min_pods", 1)
    
    action_scale_up = APP_CONFIG["actions"]["scale_up"]
    action_scale_down = APP_CONFIG["actions"]["scale_down"]
    action_restart = APP_CONFIG["actions"]["restart"]
    
    reward_ideal = APP_CONFIG["rewards"]["mock_ideal"]
    reward_waste = APP_CONFIG["rewards"]["mock_waste"]
    penalty_cpu_high = APP_CONFIG["rewards"]["mock_cpu_high_load"]
    penalty_ram_high = APP_CONFIG["rewards"]["mock_ram_high_load"]
    penalty_restart = APP_CONFIG["rewards"]["mock_restart_penalty"]
    penalty_thrashing = APP_CONFIG["rewards"].get("mock_thrashing_penalty", -500.0)
    penalty_catastrophic = APP_CONFIG["rl_hyperparameters"].get("catastrophic_penalty", -2000.0)
    
    if done:
        return penalty_catastrophic

    reward = 0.0

    if cpu_bucket == ideal_cpu and ram_bucket == ideal_ram and replicas == ideal_replicas:
        reward += reward_ideal * 1.5

    is_high_load = False
    
    if cpu_bucket >= high_threshold:
        is_high_load = True
        severity_cpu = (cpu_bucket - high_threshold) + 1
        if action != action_scale_up:
            reward += penalty_cpu_high * severity_cpu
            
    if ram_bucket >= high_threshold:
        is_high_load = True
        severity_ram = (ram_bucket - high_threshold) + 1
        if action != action_scale_up:
            reward += penalty_ram_high * severity_ram

    if is_high_load and action == action_scale_up:
        reward += reward_ideal

    if cpu_bucket <= waste_threshold and ram_bucket <= waste_threshold and replicas > min_pods:
        severity_waste = (waste_threshold - max(cpu_bucket, ram_bucket)) + 1
        
        if action == action_scale_up:
            reward += reward_waste * severity_waste * 3.0
        elif action == action_scale_down:
            reward += reward_ideal
        else:
            reward += reward_waste * severity_waste

    if action == action_restart:
        reward += penalty_restart
    
    if last_action is not None:
        is_ping_pong = False
        if action == action_scale_up and last_action == action_scale_down:
            is_ping_pong = True
        elif action == action_scale_down and last_action == action_scale_up:
            is_ping_pong = True
            
        if is_ping_pong:
            is_emergency_up = (action == action_scale_up and (cpu_bucket >= high_threshold or ram_bucket >= high_threshold))
            is_emergency_down = (action == action_scale_down and (cpu_bucket <= waste_threshold and ram_bucket <= waste_threshold))
            
            if not (is_emergency_up or is_emergency_down):
                reward += penalty_thrashing

    return reward

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
        valid_pod_states = self.max_pods - self.min_pods + 1
        
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
            
        critical_offset = APP_CONFIG["logic_constants"].get("critical_load_offset", 2)
        critical_min_pods = APP_CONFIG["logic_constants"].get("critical_min_pods", 2)
        
        if (self.cpu_bucket >= self.num_buckets - critical_offset or self.ram_bucket >= self.num_buckets - critical_offset) and self.replicas <= critical_min_pods:
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
            
        done = (self.step_count >= self.max_steps) or self.is_failure(action)

        reward = calculate_reward(self.cpu_bucket, self.ram_bucket, self.replicas, action, done)

        next_state = self._encode_state()
        info = {
            "cpu_bucket": self.cpu_bucket,
            "ram_bucket": self.ram_bucket,
            "replicas": self.replicas,
        }
        return next_state, reward, done, info