import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agents_dir)
sys.path.append(project_root)

from agents.config import RLConfig
from agents.q_learning.q_learning import QLearningAgent
from agents.q_learning.mock_env import (
    MockKubernetesEnv,
    CPU_LEVELS,
    REPLICA_LEVELS,
    ACTION_SCALE_UP,
    ACTION_SCALE_DOWN,
    ACTION_NOTHING,
    ACTION_RESTART
)

EPISODES = 200
LOG_INTERVAL = 10
INITIAL_TOTAL_REWARD = 0.0

def main():
    env = MockKubernetesEnv()

    num_states = CPU_LEVELS * REPLICA_LEVELS
    all_actions = [ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_NOTHING, ACTION_RESTART]
    num_actions = len(all_actions)

    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=RLConfig.ALPHA,
        gamma=RLConfig.GAMMA,
        epsilon=RLConfig.EPSILON,
    )

    safe_actions = [ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_NOTHING]

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = INITIAL_TOTAL_REWARD

        while not done:
            action = agent.select_action(state, allowed_actions=safe_actions)
            next_state, reward, done, info = env.step(action)

            agent.updateAction(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"Episode {episode+1}: total_reward={total_reward:.2f}")

    print("\nTraining finished")
    print("Example Q-table (first few states):")
    for s in range(num_states):
        print(f"State {s}: {agent.q_table[s]}")


if __name__ == "__main__":
    main()