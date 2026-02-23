from kazoo.client import KazooClient
import json

def setup_zookeeper_config():
    print("Connecting to ZooKeeper...")
    zk = KazooClient(hosts='127.0.0.1:2181')
    zk.start()
    print("Connected!")

    config_data = {
        "system_limits": {
            "min_pods": 1,
            "max_pods": 15,
            "replica_change_up": 1,
            "replica_change_down": -1,
            "loop_delay_seconds": 2
        },
        
        "cpu_thresholds": {
            "low": 33,
            "medium": 66
        },
        
        "rl_hyperparameters": {
            "epsilon": 0.1,         # how often to explore
            "alpha": 0.3,           # how fast the agent learns
            "gamma": 0.99,          # how much future rewards are valued
            "max_steps": 100,
            "q_value_init": 0.0,
            "catastrophic_penalty": -10.0
        },

        "rewards": {
            "good": 10.0,
            "neutral": 0.0,
            "bad": -10.0,
            "safe_reward": 0.0,
            "mock_ideal": 5.0,
            "mock_high_load": -3.0,
            "mock_waste": -2.0,
            "mock_restart_penalty": -1.0
        },

        "actions": {
            "scale_up": 0,
            "scale_down": 1,
            "no_action": 2,
            "restart": 3
        },

        "levels": {
            "low": 0,
            "medium": 1,
            "high": 2,
            "count": 3  # number of levels for state representation
        },
        "logic_constants": {
            "action_count_init": 0,
            "random_range_start": 0,
            "offset_to_last_index": 1,
            "update_factor_numerator": 1.0,
            "failure_count_init": 0,
            "failure_count_increment": 1,
            "min_tries_default": 10,
            "min_level": 0,
            "step_size": 1,
            "ideal_cpu_level": 1,
            "ideal_replicas": 1,
            "initial_cpu_level": 1,
            "initial_replicas": 1,
            "initial_step_count": 0,
            "initial_reward": 0.0,
            "min_index": 0
        }
    }

    json_data = json.dumps(config_data, indent=4).encode('utf-8')
    path = "/autoscaler/config"

    if zk.exists(path):
        zk.set(path, json_data)
        print(f"Configuration UPDATED successfully at {path}")
    else:
        zk.create(path, json_data, makepath=True)
        print(f"Configuration CREATED successfully at {path}")
    
    zk.stop()

if __name__ == "__main__":
    setup_zookeeper_config()