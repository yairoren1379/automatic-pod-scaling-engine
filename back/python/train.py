from agents.q_learning.mock_env import MockKubernetesEnv
from agents.q_learning.q_learning import QLearningAgent
from agents.bandit.bandit_safety import SafetyBandit
from config_loader import APP_CONFIG
import pickle
import matplotlib.pyplot as plt
import numpy as np

def train_system():
    max_pods = APP_CONFIG["system_limits"]["max_pods"]
    num_buckets = APP_CONFIG["metrics_config"]["num_buckets"]
    
    num_states = num_buckets * num_buckets * (max_pods + 1)
    num_actions = len(APP_CONFIG["actions"])
    
    env = MockKubernetesEnv()
    agent = QLearningAgent(num_states=num_states, num_actions=num_actions)
    
    safety_bandit = SafetyBandit(num_states=num_states, arms_count=num_actions)

    print("Start Training Session")
    NUM_EPISODES = 1000000
    
    episodes_history = []
    rewards_history = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            bandit_safe_actions = safety_bandit.get_safe_actions(state=state, max_failure_rate=0.4, min_tries=200)
            
            if not bandit_safe_actions:
                bandit_safe_actions = [APP_CONFIG["actions"]["scale_up"], APP_CONFIG["actions"]["scale_down"], APP_CONFIG["actions"]["no_action"], APP_CONFIG["actions"]["restart"]]
                
            current_pods = state % (max_pods + 1)
            final_safe_actions = bandit_safe_actions.copy()
            
            if current_pods <= 1:
                if APP_CONFIG["actions"]["scale_down"] in final_safe_actions: final_safe_actions.remove(APP_CONFIG["actions"]["scale_down"])
                if APP_CONFIG["actions"]["restart"] in final_safe_actions: final_safe_actions.remove(APP_CONFIG["actions"]["restart"])
            elif current_pods >= max_pods:
                if APP_CONFIG["actions"]["scale_up"] in final_safe_actions: final_safe_actions.remove(APP_CONFIG["actions"]["scale_up"])
            elif not final_safe_actions:
                final_safe_actions = [APP_CONFIG["actions"]["no_action"]]
                
            action = agent.select_action(state, allowed_actions=final_safe_actions)
            is_catastrophic = env.is_failure(action)
            # if is_catastrophic:
            #     print(f"[!] Catastrophic Failure detected! State: {state}, Action: {action}")
            
            next_state, reward, done, info = env.step(action)
            
            safety_bandit.update_from_outcome(state=state, action=action, is_catastrophic_failure=is_catastrophic)
            
            agent.updateAction(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Avg Reward: {total_reward:.2f}")
            episodes_history.append(episode + 1)
            rewards_history.append(total_reward)

    print("Training Finished!")
    
    # --- חלק יצירת הגרף ---
    print("Generating Learning Curve Graph...")
    plt.figure(figsize=(12, 7))

    # 1. הגרף המקורי (הקופצני) - נעשה אותו חלש וחצי שקוף
    plt.plot(episodes_history, rewards_history, color='#3b82f6', alpha=0.3, label='Raw Reward')

    # 2. הרעיון שלך: קו מגמה ליניארי (y = mx + b)
    # הפונקציה polyfit מוצאת את ה-m וה-b האידיאליים ביותר לנתונים שלנו
    m, b = np.polyfit(episodes_history, rewards_history, 1)
    trendline = [m * x + b for x in episodes_history]
    plt.plot(episodes_history, trendline, color='#10b981', linewidth=3, linestyle='--', label=f'Linear Trend (y={m:.2f}x+{b:.0f})')

    # 3. ממוצע נע (Moving Average) - מחליק את הגרף כדי לראות את ההתכנסות
    window = 15 # ממוצע של כל 15 נקודות (1500 אפיזודות)
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(episodes_history[window-1:], moving_avg, color='#ef4444', linewidth=2.5, label=f'Moving Average (Window: {window})')

    plt.title('RL Agent Learning Curve (Reward over Episodes)', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11) # מוסיף מקרא שמסביר מה זה כל קו
    plt.tight_layout()
    plt.savefig('api/learning_curve.png', dpi=300) # שומר ברזולוציה גבוהה לספר
    print("Graph saved to api/learning_curve.png")
    # -----------------------
    
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
        f.write(f"Total Episodes: {NUM_EPISODES}\n")
        f.write("-----------------------------\n\n")
        
        for state_idx, q_values in enumerate(agent.q_table):
            replicas = state_idx % (max_pods + 1)
            remaining = state_idx // (max_pods + 1)
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