class RLConfig:
    # Hyperparameters
    EPSILON = 0.1 # הסתברות לחקירה how often to explore
    ALPHA = 0.1 # מקדם למידה how fast the agent learns
    GAMMA = 0.99 # פקטור הנחה how much future rewards are valued
    
    # Rewards & Penalties
    CATASTROPHIC_PENALTY = -100.0
    SAFE_REWARD = 0.0
    
    # Environment Settings
    MAX_STEPS = 100
    Q_VALUE_INIT = 0.0