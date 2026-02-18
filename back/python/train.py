from agents.q_learning.mock_env import MockKubernetesEnv
from agents.q_learning.q_learning import QLearningAgent
from agents.bandit.bandit_safety import SafetyBandit
import pickle

def train_system():
    env = MockKubernetesEnv()
    agent = QLearningAgent(num_states=9, num_actions=4)
    safety_bandit = SafetyBandit(arms_count=4)

    print("Start Training Session")

    for episode in range(100000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            safe_actions = safety_bandit.get_safe_actions(max_failure_rate=0.4, min_tries=7000)
            
            if not safe_actions:
                # [ScaleUp, ScaleDown, None, Restart]
                safe_actions = [0, 1, 2, 3]
                
            action = agent.select_action(state, allowed_actions=safe_actions)
            is_catastrophic = env.is_failure(action)
            next_state, reward, done, info = env.step(action)
            safety_bandit.update_from_outcome(action, is_catastrophic)
            agent.updateAction(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Avg Reward: {total_reward:.2f}")

    print("Training Finished!")
    
    with open("api/brain_model.pkl", "wb") as f:
        pickle.dump({
            "q_table": agent.q_table,
            "bandit_counts": safety_bandit.action_counts,
            "bandit_failures": safety_bandit.failure_counts
        }, f)
    
    print("Model saved to brain_model.pkl")

    action_names = {0: "ScaleUp", 1: "ScaleDown", 2: "None", 3: "Restart"}
    
    with open("api/brain_readable.txt", "w") as f:
        f.write("--- Q-Learning Final Report ---\n")
        f.write(f"Total Episodes: 100000\n")
        f.write("-----------------------------\n\n")
        
        for state_idx, q_values in enumerate(agent.q_table):
            cpu_level = state_idx // 3
            replicas = (state_idx % 3) + 1
            
            f.write(f"State {state_idx} [CPU Level: {cpu_level}, Pods: {replicas}]:\n")
            
            for action_idx, score in enumerate(q_values):
                action_name = action_names.get(action_idx, "Unknown")
                f.write(f"  Action '{action_name}': {score:.2f}\n")
            
            best_action_idx = q_values.index(max(q_values))
            best_action_name = action_names[best_action_idx]
            f.write(f"  >> BEST CHOICE: {best_action_name}\n")
            f.write("-----------------------------\n")
            
    print("Readable report saved to brain_readable.txt")

    
    return agent, safety_bandit

if __name__ == "__main__":
    train_system()