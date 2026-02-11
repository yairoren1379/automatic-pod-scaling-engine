from agents.q_learning.mock_env import MockKubernetesEnv
from agents.q_learning.q_learning import QLearningAgent
from agents.bandit.bandit_safety import SafetyBandit

def train_system():
    env = MockKubernetesEnv()
    agent = QLearningAgent(num_states=9, num_actions=4)
    safety_bandit = SafetyBandit(arms_count=4)

    print("Start Training Session")

    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            safe_actions = safety_bandit.get_safe_actions(max_failure_rate=0.2, min_tries=5)
            
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

    print("--- Training Finished! ---")
    
    return agent, safety_bandit

if __name__ == "__main__":
    train_system()