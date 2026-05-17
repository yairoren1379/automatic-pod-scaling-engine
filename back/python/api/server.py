import sys
import os
import json
import subprocess
import pickle
import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# in order to import from agents module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config_loader import APP_CONFIG
from agents.q_learning.q_learning import QLearningAgent

from history_db import log_run

app = FastAPI(title="K8s RL Learning Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_system_status = {
    "pods": 0, "cpu_usage": 0.0, "ram_usage": 0.0, "cpu_bucket": 0, "ram_bucket": 0,
    "action": "Waiting...", "reward": 0.0, "q_values": [0,0,0,0]
}

brain_logs_buffer = []

def add_log(msg: str):
    print(msg)
    brain_logs_buffer.append(msg)
    if len(brain_logs_buffer) > 100:
        brain_logs_buffer.pop(0)

@app.get("/status")
def get_dashboard_status():
    return last_system_status

@app.get("/logs-data")
def get_logs_data():
    return {"logs": brain_logs_buffer}

system_resting = False

def apply_system_rest():
    global system_resting
    system_resting = True
    add_log("\n[SYSTEM] Entering 30 seconds cooldown period...")
    time.sleep(30)
    system_resting = False
    add_log("[SYSTEM] Cooldown finished. AI is awake.\n")

MAX_PODS = APP_CONFIG.get("system_limits", {}).get("max_pods", 15)
NUM_BUCKETS = APP_CONFIG.get("metrics_config", {}).get("num_buckets", 34)

num_states = NUM_BUCKETS * NUM_BUCKETS * (MAX_PODS + 1)

agent = QLearningAgent(num_states=num_states, num_actions=len(APP_CONFIG["actions"]))

if os.path.exists("brain_model.pkl"):
    with open("brain_model.pkl", "rb") as f:
        data = pickle.load(f)
        agent.q_table = data["q_table"]
        agent.epsilon = 0.05
    print("Loaded pre-trained model successfully!")
else:
    print("No pre-trained model found. Starting with fresh agent.")
    
class ClusterState(BaseModel):
    pod_count: int
    cpu_usage: float
    ram_usage: float
    is_crashed: bool

def get_bucket(usage: float) -> int:
    return int(max(0, min(100, usage)) // APP_CONFIG.get("metrics_config", {}).get("bucket_step", 3))

def encode_state(cpu_bucket: int, ram_bucket: int, replicas: int) -> int:
    return (cpu_bucket * NUM_BUCKETS * (MAX_PODS + 1)) + (ram_bucket * (MAX_PODS + 1)) + replicas

def get_action_string(action_id: int) -> str:
    mapping = {
        APP_CONFIG["actions"]["scale_up"]: "ScaleUp",
        APP_CONFIG["actions"]["scale_down"]: "ScaleDown",
        APP_CONFIG["actions"]["no_action"]: "None",
        APP_CONFIG["actions"]["restart"]: "Restart"
    }
    return mapping.get(action_id, "None")

class StateRequest(BaseModel):
    cpu_percentage: float
    ram_percentage: float
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
    return {"status": "Learning Engine is Running"}

@app.post("/decide")
def decide(req: ClusterState):
    global system_resting
    if system_resting:
        last_system_status["action"] = "Resting (30s)..."
        return {"action": "Resting"}

    cpu_bucket = get_bucket(req.cpu_usage)
    ram_bucket = get_bucket(req.ram_usage)
    current_replicas = min(req.pod_count, MAX_PODS)
    
    state_idx = encode_state(cpu_bucket, ram_bucket, current_replicas)
    
    safe_actions = list(APP_CONFIG["actions"].values())
    action_id = agent.select_action(state_idx, allowed_actions=safe_actions)
    action_str = get_action_string(action_id)
    
    last_system_status["pods"] = current_replicas
    last_system_status["cpu_usage"] = req.cpu_usage
    last_system_status["ram_usage"] = req.ram_usage
    last_system_status["cpu_bucket"] = cpu_bucket
    last_system_status["ram_bucket"] = ram_bucket
    last_system_status["action"] = action_str
    last_system_status["q_values"] = agent.q_table[state_idx]
    
    return {"action": action_str}

@app.post("/predict")
def get_action(req: StateRequest):
    global system_resting
    if system_resting:
        return {
            "recommended_action": APP_CONFIG["actions"]["no_action"],
            "state_index": 0,
            "action_string": "Resting...",
            "q_values": [0,0,0,0]
        }

    cpu_bucket = get_bucket(req.cpu_percentage)
    ram_bucket = get_bucket(req.ram_percentage)
    state_idx = encode_state(cpu_bucket, ram_bucket, req.replicas)
    
    if state_idx >= num_states or state_idx < APP_CONFIG["logic_constants"]["min_index"]:
        raise HTTPException(status_code=400, detail="State out of bounds")
        
    action = agent.select_action(state_idx, allowed_actions=req.allowed_actions)
    
    return {
        "recommended_action": action,
        "state_index": state_idx,
        "action_string": get_action_string(action),
        "q_values": agent.q_table[state_idx]
    }

def apply_k8s_patch(command_list):
    patch = {"spec": {"template": {"spec": {"containers": [{"name": "python-container", "command": command_list}]}}}}
    
    patch_file_path = os.path.join(current_dir, "patch.json")
    with open(patch_file_path, "w") as f:
        json.dump(patch, f)
    
    cmd = f'kubectl patch deployment yair-api-python --patch-file "{patch_file_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        add_log("[SYSTEM] Load command applied! K8s is recreating pods (wait up to 60s for metrics server).")
    else:
        add_log(f"[ERROR] Failed to patch K8s: {result.stderr}")

@app.post("/start-load")
def start_load(background_tasks: BackgroundTasks):
    apply_k8s_patch(["/bin/sh", "-c", "while true; do true; done"])
    background_tasks.add_task(apply_system_rest)
    return {"status": "High Load Started, entering cooldown"}

@app.post("/stop-load")
def stop_load(background_tasks: BackgroundTasks):
    apply_k8s_patch(["/bin/sh", "-c", "sleep 3600"])
    background_tasks.add_task(apply_system_rest)
    return {"status": "Load Stopped, entering cooldown"}

@app.post("/scale-min")
def scale_min(background_tasks: BackgroundTasks):
    cmd = "kubectl scale deployment yair-api-python --replicas=0"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        add_log("[SYSTEM] Scaled K8s deployment to 0 pods!")
    else:
        add_log(f"[ERROR] Failed to scale to 0: {result.stderr}")
        
    background_tasks.add_task(apply_system_rest)
    return {"status": "Scaled to 0, entering cooldown"}

@app.post("/scale-max")
def scale_max(background_tasks: BackgroundTasks):
    cmd = "kubectl scale deployment yair-api-python --replicas=15"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        add_log("[SYSTEM] Scaled K8s deployment to 15 pods!")
    else:
        add_log(f"[ERROR] Failed to scale to 15: {result.stderr}")
        
    background_tasks.add_task(apply_system_rest)
    return {"status": "Scaled to 15, entering cooldown"}

step_counter = 0

@app.post("/train")
def update_agent(req: LearnRequest):
    global step_counter, system_resting
    if system_resting:
        return {"status": "resting, skipped training"}
    
    current_replicas = min(req.state.replicas, MAX_PODS)
    next_replicas = min(req.next_state.replicas, MAX_PODS)
    
    cpu_bucket = get_bucket(req.state.cpu_percentage)
    ram_bucket = get_bucket(req.state.ram_percentage)
    state_idx = encode_state(cpu_bucket, ram_bucket, current_replicas)
    
    next_cpu_bucket = get_bucket(req.next_state.cpu_percentage)
    next_ram_bucket = get_bucket(req.next_state.ram_percentage)
    next_state_idx = encode_state(next_cpu_bucket, next_ram_bucket, next_replicas)

    agent.updateAction(state=state_idx, action=req.action, reward=req.reward, next_state=next_state_idx, done=req.done)
    last_system_status["reward"] = req.reward
    
    is_catastrophic = req.reward <= APP_CONFIG["rl_hyperparameters"].get("catastrophic_penalty", -10.0)
    new_q_val = agent.q_table[state_idx][req.action]
    
    try:
        log_run(state=state_idx, action=req.action, reward=req.reward, is_catastrophic=is_catastrophic, new_q_value=new_q_val)
    except Exception as e:
        add_log(f"[DB ERROR] Failed to save to database: {e}")
    
    step_counter += 1
    if step_counter % 2 == 0:
        q_values = agent.q_table[state_idx]
        log_text = (
            f"--- Q-Table Snapshot (Step {step_counter}) ---\n"
            f"State: [CPU:{req.state.cpu_percentage}% RAM:{req.state.ram_percentage}% Pods:{current_replicas}] | Action: {get_action_string(req.action)} | Reward: {req.reward}\n"
            f"Brain Knowledge -> ScaleUp: {q_values[APP_CONFIG['actions']['scale_up']]:.2f} | "
            f"ScaleDown: {q_values[APP_CONFIG['actions']['scale_down']]:.2f} | "
            f"None: {q_values[APP_CONFIG['actions']['no_action']]:.2f}\n"
        )
        add_log(log_text)

    return {"status": "updated", "new_q_value": agent.q_table[state_idx][req.action]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)