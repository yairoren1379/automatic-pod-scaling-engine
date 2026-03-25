import sys
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pickle

# in order to import from agents module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config_loader import APP_CONFIG
from agents.q_learning.q_learning import QLearningAgent
from agents.bandit.bandit_safety import SafetyBandit

app = FastAPI(title="K8s RL Learning Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_system_status = {
    "pods": 0, "cpu_usage": 0.0, "cpu_level": 0,
    "action": "Waiting...", "reward": 0.0, "q_values": [0,0,0,0]
}

@app.get("/status")
def get_dashboard_status():
    return last_system_status

MAX_PODS = APP_CONFIG.get("system_limits", {}).get("max_pods", 15)
num_states = APP_CONFIG["levels"]["count"] * (MAX_PODS + 1)

# [scale up = 0, scale down = 1, nothing = 2, restart = 3]
agent = QLearningAgent(
    num_states=num_states,
    num_actions=len(APP_CONFIG["actions"])
)

if os.path.exists("brain_model.pkl"):
    with open("brain_model.pkl", "rb") as f:
        data = pickle.load(f)
        agent.q_table = data["q_table"]
    print("Loaded pre-trained model successfully!")
else:
    print("No pre-trained model found. Starting with fresh agent.")

class ClusterState(BaseModel):
    pod_count: int
    cpu_usage: float # in percentage
    is_crashed: bool #are any pods crashed

def get_cpu_level(usage: float) -> int:
    if usage < APP_CONFIG["cpu_thresholds"]["low"]:
        return APP_CONFIG["levels"]["low"]
    elif usage < APP_CONFIG["cpu_thresholds"]["medium"]:
        return APP_CONFIG["levels"]["medium"]
    else:
        return APP_CONFIG["levels"]["high"]

def get_action_string(action_id: int) -> str:
    mapping = {
        APP_CONFIG["actions"]["scale_up"]: "ScaleUp",
        APP_CONFIG["actions"]["scale_down"]: "ScaleDown",
        APP_CONFIG["actions"]["no_action"]: "None",
        APP_CONFIG["actions"]["restart"]: "Restart"
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
    current_replicas = min(req.pod_count, MAX_PODS)
    state_idx = cpu_level * (MAX_PODS + 1) + current_replicas
    safe_actions = list(APP_CONFIG["actions"].values())
    action_id = agent.select_action(state_idx, allowed_actions=safe_actions)
    action_str = get_action_string(action_id)
    
    last_system_status["pods"] = current_replicas
    last_system_status["cpu_usage"] = req.cpu_usage
    last_system_status["cpu_level"] = cpu_level
    last_system_status["action"] = action_str
    last_system_status["q_values"] = agent.q_table[state_idx]
    
    return {"action": action_str}

@app.post("/predict")
# gets the current state and returns the recommended action by the agent
def get_action(req: StateRequest):
    # convert the state to an index in base 3
    state_idx = req.cpu_level * (MAX_PODS + 1) + req.replicas
    
    # In case of state that is out of bounds
    if state_idx >= num_states or state_idx < APP_CONFIG["logic_constants"]["min_index"]:
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
    
    current_replicas = min(req.state.replicas, MAX_PODS)
    next_replicas = min(req.next_state.replicas, MAX_PODS)
    
    # convert states to indexes in base of REPLICA_LEVELS value
    state_idx = req.state.cpu_level * (MAX_PODS + 1) + current_replicas
    next_state_idx = req.next_state.cpu_level * (MAX_PODS + 1) + next_replicas

    # updates the agent with the new experience
    agent.updateAction(
        state=state_idx,
        action=req.action,
        reward=req.reward,
        next_state=next_state_idx,
        done=req.done
    )
    
    last_system_status["reward"] = req.reward
    
    step_counter += 1
    if step_counter % 2 == 0: # print every 2 steps
        print(f"\n--- Q-Table Snapshot (Step {step_counter}) ---")
        print(f"Current State [CPU:{req.state.cpu_level}, Pods:{current_replicas}] Index: {state_idx}")
        print(f"Action Taken: {get_action_string(req.action)} | Reward: {req.reward}")
        
        q_values = agent.q_table[state_idx]
        print(f"Knowledge for this state:")
        print(f"  ScaleUp:   {q_values[APP_CONFIG['actions']['scale_up']]:.2f}")
        print(f"  ScaleDown: {q_values[APP_CONFIG['actions']['scale_down']]:.2f}")
        print(f"  None:      {q_values[APP_CONFIG['actions']['no_action']]:.2f}")
        print(f"  Restart:   {q_values[APP_CONFIG['actions']['restart']]:.2f}")
        print("------------------------------------------\n")

    return {"status": "updated", "new_q_value": agent.q_table[state_idx][req.action]}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)