import sys
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# in order to import from agents module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from agents.config import RLConfig
from agents.q_learning.q_learning import QLearningAgent
from agents.q_learning.mock_env import CPU_LEVELS, REPLICA_LEVELS
from agents.bandit.bandit_safety import SafetyBandit

from config import LOW_CPU, MEDIUM_CPU, LOW_LEVEL, MEDIUM_LEVEL, HIGH_LEVEL

app = FastAPI(title="K8s RL Learning Engine")
NUM_ACTIONS = 4
MIN_INDEX = 0

num_states = CPU_LEVELS * REPLICA_LEVELS
# [scale up = 0, scale down = 1, nothing = 2, restart = 3]

agent = QLearningAgent(
    num_states=num_states,
    num_actions= NUM_ACTIONS
)

safety_bandit = SafetyBandit(arms_count=NUM_ACTIONS)

class ClusterState(BaseModel):
    pod_count: int
    cpu_usage: float # in percentage
    is_crashed: bool #are any pods crashed

def get_cpu_level(usage: float) -> int:
    if usage < LOW_CPU:
        return LOW_LEVEL
    elif usage < MEDIUM_CPU:
        return MEDIUM_LEVEL
    else:
        return HIGH_LEVEL

def get_action_string(action_id: int) -> str:
    mapping = {
        0: "ScaleUp",
        1: "ScaleDown",
        2: "None",
        3: "Restart"
    }
    return mapping.get(action_id, "None")

class StateRequest(BaseModel):
    cpu_level: int
    replicas: int
    allowed_actions: Optional[List[int]] = None
    
class LearnRequest(BaseModel):
    state: StateRequest
    action: int
    reward: float
    next_state: StateRequest
    done: bool

@app.get("/")
def read_root():
    return {"status": "Learning Engine is Running", "agent": str(agent)}

@app.post("/decide")
def decide(req: ClusterState):
    cpu_level = get_cpu_level(req.cpu_usage)
    current_replicas = min(req.pod_count, REPLICA_LEVELS - 1)
    state_idx = cpu_level * REPLICA_LEVELS + current_replicas
    safe_actions = safety_bandit.get_safe_actions(max_failure_rate=0.2)
    action_id = agent.select_action(state_idx, allowed_actions=safe_actions)
    action_str = get_action_string(action_id)
    return {"action": action_str}

@app.post("/predict")
# gets the current state and returns the recommended action by the agent
def get_action(req: StateRequest):
    # convert the state to an index in base 3
    state_idx = req.cpu_level * REPLICA_LEVELS + req.replicas
    
    # In case of state that is out of bounds
    if state_idx >= num_states or state_idx < MIN_INDEX:
        #code 400 - client error
        raise HTTPException(status_code=400, detail="State out of bounds")

    # choose the action
    action = agent.select_action(state_idx, allowed_actions=req.allowed_actions)
    
    return {
        "recommended_action": action,
        "state_index": state_idx,
        "action_string": get_action_string(action),
        "q_values": agent.q_table[state_idx]
    }

step_counter = 0

@app.post("/train")
def update_agent(req: LearnRequest):
    global step_counter
    
    current_replicas = min(req.state.replicas, REPLICA_LEVELS - 1)
    next_replicas = min(req.next_state.replicas, REPLICA_LEVELS - 1)
    
    # convert states to indexes in base of REPLICA_LEVELS value
    state_idx = req.state.cpu_level * REPLICA_LEVELS + current_replicas
    next_state_idx = req.next_state.cpu_level * REPLICA_LEVELS + next_replicas

    # updates the agent with the new experience
    agent.updateAction(
        state=state_idx,
        action=req.action,
        reward=req.reward,
        next_state=next_state_idx,
        done=req.done
    )
    
    step_counter += 1
    if step_counter % 1 == 5: # מדפיס כל 5 אימונים
        print(f"\n--- Q-Table Snapshot (Step {step_counter}) ---")
        print(f"Current State [CPU:{req.state.cpu_level}, Pods:{current_replicas}] Index: {state_idx}")
        print(f"Action Taken: {get_action_string(req.action)} | Reward: {req.reward}")
        
        # הדפסת הערכים של המצב הנוכחי בלבד
        q_values = agent.q_table[state_idx]
        print(f"Knowledge for this state:")
        print(f"  ScaleUp:   {q_values[0]:.2f}")
        print(f"  ScaleDown: {q_values[1]:.2f}")
        print(f"  None:      {q_values[2]:.2f}")
        print(f"  Restart:   {q_values[3]:.2f}")
        print("------------------------------------------\n")

    return {"status": "updated", "new_q_value": agent.q_table[state_idx][req.action]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)