# Legacy: the first prototype (tabular Q-learning + safety bandit)

This folder keeps the project's first stage for reference. It is **not used** by the system.

- `q_learning/q_learning.py` - a Q-table with one row per (CPU bucket, RAM bucket, replicas).
- `bandit/bandit_safety.py` - a multi-armed bandit that counted catastrophic failures per action.
- `q_learning/mock_env.py` - a random-walk environment with a reward based on action labels.
- `train.py`, `server.py`, `brain_model.pkl` - training script, API and the trained table.

Why it was replaced by the DQN (see the root README):

1. A table can't use continuous metrics or forecasts: every new input multiplies the number of
   rows. A neural network generalizes between similar states.
2. The old reward rewarded action labels ("scale up under high load is good"), so the agent
   learned to repeat rules instead of outcomes. The new reward comes from the resulting state.
3. The old environment was a random walk, with no cause and effect for a forecaster to learn.

The code ran against the old configuration format (git history, commit `215f06d`) and is not
runnable with the current `setup_config.py`.
