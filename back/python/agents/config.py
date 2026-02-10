class RLConfig:
    # Hyperparameters
    EPSILON = 0.1
    ALPHA = 0.1
    GAMMA = 0.99
    
    # Rewards & Penalties
    CATASTROPHIC_PENALTY = -100.0
    SAFE_REWARD = 0.0
    
    # Environment Settings
    MAX_STEPS = 100
    Q_VALUE_INIT = 0.0