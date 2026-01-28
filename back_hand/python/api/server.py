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

app = FastAPI(title="K8s RL Learning Engine")
NUM_ACTIONS = 4
MIN_INDEX = 0

num_states = CPU_LEVELS * REPLICA_LEVELS
# [scale up = 0, scale down = 1, nothing = 2, restart = 3]

agent = QLearningAgent(
    num_states=num_states,
    num_actions= NUM_ACTIONS
)

# to send the controller the states and allowed actions
class StateRequest(BaseModel):
    cpu_level: int
    replicas: int
    allowed_actions: Optional[List[int]] = None

# to send the controller the learning feedback
class LearnRequest(BaseModel):
    state: StateRequest
    action: int
    reward: float
    next_state: StateRequest
    done: bool

@app.get("/")
def read_root():
    return {"status": "Learning Engine is Running", "agent": str(agent)}

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
        "q_values": agent.q_table[state_idx]# returning the Q-values for the given state
    }

@app.post("/train")
def update_agent(req: LearnRequest):
    # convert states to indexes in base of REPLICA_LEVELS value
    state_idx = req.state.cpu_level * REPLICA_LEVELS + req.state.replicas
    next_state_idx = req.next_state.cpu_level * REPLICA_LEVELS + req.next_state.replicas

    # updates the agent with the new experience
    agent.updateAction(
        state=state_idx,
        action=req.action,
        reward=req.reward,
        next_state=next_state_idx,
        done=req.done
    )

    return {"status": "updated", "new_q_value": agent.q_table[state_idx][req.action]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)