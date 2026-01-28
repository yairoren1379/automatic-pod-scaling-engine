import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agents_dir)
sys.path.append(project_root)

from bandit_safety import SafetyBandit

EPISODES = 10000
LOG_INTERVAL = 1000
MAX_FAILURE_RATE = 0.05

SCALE_UP = 0.05
SCALE_DOWN = 0.10
NOTHING = 0.01
RESTART = 0.30

def main():
    failure_probs = [SCALE_UP, SCALE_DOWN, NOTHING, RESTART]

    bandit = SafetyBandit(
        arms_count=len(failure_probs))

    for t in range(EPISODES):
        action = bandit.select_action()
        is_failure = random.random() < failure_probs[action]
        bandit.update_from_outcome(action, is_failure)

        if (t + 1) % LOG_INTERVAL == 0:
            print(f"Step {t+1}")
            print("failures:", bandit.failure_counts)
            print("safe actions:", bandit.get_safe_actions(max_failure_rate=MAX_FAILURE_RATE))

    print("\nFinal:")
    print("Failure counts:", bandit.failure_counts)
    print("Safe actions:", bandit.get_safe_actions(max_failure_rate=MAX_FAILURE_RATE))

if __name__ == "__main__":
    main()
