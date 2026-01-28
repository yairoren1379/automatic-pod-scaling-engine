from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ClusterState(BaseModel):
    pod_count: int  #in precentage
    cpu_usage: float
    is_crashed: bool

@app.post("/decide")
async def get_action(state: ClusterState):
    print(f" Brain received state: {state}")
        
    action = "do_nothing"
    
    if state.is_crashed:
        action = "scale_up"
        print(" CRASH DETECTED! Ordering Scale Up.")
    elif state.cpu_usage > 80:
        action = "scale_up"
        print(" High Load! Ordering Scale Up.")
    
    return {"action": action}