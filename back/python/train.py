from agents.q_learning.mock_env import MockKubernetesEnv
from agents.q_learning.q_learning import QLearningAgent
from agents.bandit.bandit_safety import SafetyBandit
from config_loader import APP_CONFIG
import pickle
import matplotlib.pyplot as plt
import numpy as np

def train_system():
    min_pods = APP_CONFIG["system_limits"]["min_pods"]
    max_pods = APP_CONFIG["system_limits"]["max_pods"]
    num_buckets = APP_CONFIG["metrics_config"]["num_buckets"]
    valid_pod_states = max_pods - min_pods + 1
    
    num_states = num_buckets * num_buckets * valid_pod_states
    num_actions = len(APP_CONFIG["actions"])
    
    env = MockKubernetesEnv()
    agent = QLearningAgent(num_states=num_states, num_actions=num_actions)
    
    all_possible_actions = list(APP_CONFIG["actions"].values())

    for state_idx in range(num_states):
        current_pods = (state_idx % valid_pod_states) + min_pods
        allowed_for_this_state = all_possible_actions.copy()
        
        if current_pods <= min_pods and APP_CONFIG["actions"]["scale_down"] in allowed_for_this_state:
            allowed_for_this_state.remove(APP_CONFIG["actions"]["scale_down"])
            
        if current_pods >= max_pods and APP_CONFIG["actions"]["scale_up"] in allowed_for_this_state:
            allowed_for_this_state.remove(APP_CONFIG["actions"]["scale_up"])
            
        for action in all_possible_actions:
            if action not in allowed_for_this_state:
                agent.q_table[state_idx][action] = -1e9
                
    safety_bandit = SafetyBandit(num_states=num_states, arms_count=num_actions)

    print("Start Training Session")
    
    alpha_val = APP_CONFIG["rl_hyperparameters"]["alpha"]
    
    episodes_history = []
    rewards_history = []
    count_with_epsilon_above_min = 0

    recent_rewards = []
    window_size = 1000
    
    convergence_threshold = APP_CONFIG["rl_hyperparameters"]["convergence_threshold"]
    print(f"Dynamic Convergence Threshold set to: {convergence_threshold:.3f} (based on Alpha: {alpha_val})")
    
    previous_window_avg = None
    reward_diff = float('inf')

    episode = 0
    while reward_diff > convergence_threshold:
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            bandit_safe_actions = safety_bandit.get_safe_actions(state=state, max_failure_rate=0.4, min_tries=200)
            
            if not bandit_safe_actions:
                bandit_safe_actions = [APP_CONFIG["actions"]["scale_up"], APP_CONFIG["actions"]["scale_down"], APP_CONFIG["actions"]["no_action"], APP_CONFIG["actions"]["restart"]]
                
            current_pods = (state % valid_pod_states) + min_pods
            final_safe_actions = bandit_safe_actions.copy()
            
            if current_pods <= min_pods:
                if APP_CONFIG["actions"]["scale_down"] in final_safe_actions: final_safe_actions.remove(APP_CONFIG["actions"]["scale_down"])
            if current_pods >= max_pods:
                if APP_CONFIG["actions"]["scale_up"] in final_safe_actions: final_safe_actions.remove(APP_CONFIG["actions"]["scale_up"])
            
            action = agent.select_action(state, allowed_actions=final_safe_actions)
            is_catastrophic = env.is_failure(action)
            next_state, reward, done, info = env.step(action)
            
            if is_catastrophic:
                reward += APP_CONFIG["rl_hyperparameters"]["catastrophic_penalty"]
                done = True
            
            safety_bandit.update_from_outcome(state=state, action=action, is_catastrophic_failure=is_catastrophic)
            agent.updateAction(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        if(agent.epsilon > APP_CONFIG["rl_hyperparameters"]["epsilon_min"]):
            count_with_epsilon_above_min += 1
            
        rewards_history.append(total_reward)
        episodes_history.append(episode + 1)
        
        recent_rewards.append(total_reward)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Avg Reward: {total_reward:.2f}")

        if (episode + 1) % 5000 == 0 and len(recent_rewards) == window_size:
            current_window_avg = np.mean(recent_rewards)
            
            if previous_window_avg is not None:
                reward_diff = abs(current_window_avg - previous_window_avg)
                print(f"--- Episode {episode+1}: Current Avg: {current_window_avg:.2f}, Prev Avg: {previous_window_avg:.2f}, Diff: {reward_diff:.4f} ---")
                    
            previous_window_avg = current_window_avg
            
        episode += 1

    print("Training Finished!")
    print("------------------------------------")
    print("epsilon:", agent.epsilon)
    print("Total episodes ran:", episode + 1)
    print("Episodes with epsilon > min:", count_with_epsilon_above_min)
    print("Episodes with epsilon < min:", (episode + 1) - count_with_epsilon_above_min)
    print("------------------------------------")

    gamma_val = APP_CONFIG["rl_hyperparameters"]["gamma"]
    decay_val = APP_CONFIG["rl_hyperparameters"]["epsilon_decay"]
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_history, rewards_history, alpha=0.3, label='Raw Reward')
    
    window = 50
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(episodes_history[window-1:], moving_avg, color='red', label='Moving Avg')
    
    plt.title(f'Learning Curve\nAlpha: {alpha_val} | Gamma: {gamma_val} | Epsilon Decay: {decay_val}\nStop Diff: {convergence_threshold:.3f}')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('api/learning_curve.png')
    
    with open("api/brain_model.pkl", "wb") as f:
        pickle.dump({
            "q_table": agent.q_table,
            "bandit_counts": safety_bandit.action_counts,
            "bandit_failures": safety_bandit.failure_counts
        }, f)
    
    print("Model saved to brain_model.pkl")
    
    action_names = {v: k for k, v in APP_CONFIG["actions"].items()}
    with open("api/brain_readable.txt", "w") as f:
        f.write("--- Q-Learning Final Report ---\n")
        f.write(f"Total Episodes: {episode + 1}\n")
        f.write("-----------------------------\n\n")
        
        for state_idx, q_values in enumerate(agent.q_table):
            replicas = (state_idx % valid_pod_states) + min_pods
            remaining = state_idx // valid_pod_states
            ram_bucket = remaining % num_buckets
            cpu_bucket = remaining // num_buckets
            
            f.write(f"State {state_idx} [CPU Bucket: {cpu_bucket}, RAM Bucket: {ram_bucket}, Pods: {replicas}]:\n")
            for action_idx, score in enumerate(q_values):
                action_name = action_names.get(action_idx, "Unknown")
                f.write(f"  Action '{action_name}': {score:.2f}\n")
            
            best_action_idx = q_values.index(max(q_values))
            best_action_name = action_names.get(best_action_idx, "Unknown")
            f.write(f"  >> BEST CHOICE: {best_action_name}\n")
            f.write("-----------------------------\n")
            
    print("Readable report saved to brain_readable.txt")

    return agent, safety_bandit

if __name__ == "__main__":
    train_system()