"""The learning engine's REST + WebSocket API (FastAPI).

Run from back/python:
    python -m api.server            (or: uvicorn api.server:app --port 8000)

For the Go controller:
    POST /api/decide        metrics in, DQN decision out
    POST /api/report        what was executed and what the guardrails did

For the dashboard:
    WS   /ws                live stream: every control step, audit entry, config and load change
    GET  /api/status        current snapshot
    GET  /api/history       recent metrics time series
    GET  /api/audit         recent decisions vs guardrail interventions
    GET  /api/online-training   online learning curve (reward, loss, epsilon)
    GET  /api/runs, /api/runs/{name}, /api/experiments, /api/models   offline training results
    GET/PUT /api/config     configuration (stored in ZooKeeper)
    POST /api/mode          baseline | shadow | active
    GET/POST /api/load      load profile for the simulation panel
"""
import asyncio
import copy
import csv
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)   # allow running as a script from any folder

from core.state import ClusterMetrics                                          # noqa: E402
from engine.learning_engine import MODES, LearningEngine                      # noqa: E402
from engine.zk_store import ZkStore                                            # noqa: E402
from setup_config import CONFIG_PATH, DEFAULT_CONFIG, DEFAULT_LOAD, LOAD_PATH, ZK_HOSTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
DASHBOARD_BUILD = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "front", "build")


# ---------------------------------------------------------------------------------------------
# WebSocket broadcasting
# ---------------------------------------------------------------------------------------------

class Broadcaster:
    """Pushes events to every connected dashboard (no polling)."""

    def __init__(self):
        self.clients = set()
        self.loop = None

    async def send(self, event: dict):
        for client in list(self.clients):
            try:
                await client.send_json(event)
            except Exception:
                self.clients.discard(client)

    def send_threadsafe(self, event: dict):
        """For callbacks that run outside the event loop (ZooKeeper watches)."""
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.send(event), self.loop)


broadcaster = Broadcaster()
state = {}   # filled at startup: store, engine, config, load


def on_config_change(config: dict):
    merged = merge_with_defaults(config)
    state["config"] = merged
    if "engine" in state:
        state["engine"].apply_config(merged)
    broadcaster.send_threadsafe({"type": "config", "config": merged})


def on_load_change(load: dict):
    state["load"] = load
    broadcaster.send_threadsafe({"type": "load", "load": load})


def merge_with_defaults(config: dict) -> dict:
    """A ZNode written by an older version may miss new keys: fill them from the defaults."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for section, values in config.items():
        if isinstance(values, dict) and section in merged:
            merged[section].update(values)
    return merged


@asynccontextmanager
async def lifespan(app: FastAPI):
    broadcaster.loop = asyncio.get_running_loop()
    store = ZkStore(ZK_HOSTS, {CONFIG_PATH: DEFAULT_CONFIG, LOAD_PATH: DEFAULT_LOAD},
                    enabled=os.environ.get("ZK_DISABLED") != "1")
    state["store"] = store
    state["config"] = merge_with_defaults(store.get(CONFIG_PATH))
    state["load"] = store.get(LOAD_PATH)
    engine = LearningEngine(state["config"], MODELS_DIR, DATA_DIR)
    state["engine"] = engine
    store.watch(CONFIG_PATH, on_config_change)
    store.watch(LOAD_PATH, on_load_change)
    log.info("Model: %s | forecaster: %s | ZooKeeper: %s", engine.model_source, engine.forecaster_source,
             "connected" if store.connected else "offline (in-memory config)")
    yield
    engine.save()
    store.close()


app = FastAPI(title="Autoscaler Learning Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------------------------
# Go controller endpoints
# ---------------------------------------------------------------------------------------------

class MetricsIn(BaseModel):
    cpu_percent: float
    ram_percent: float
    latency_ms: float
    qps: float
    error_rate_percent: float
    replicas: int
    pod_crashed: bool = False


class DecideRequest(BaseModel):
    mode: str
    metrics: MetricsIn


class ReportRequest(BaseModel):
    decision_id: int
    executed_action: str            # ScaleUp | ScaleDown | Hold
    source: str                     # "dqn" or "baseline": who chose the executed action
    blocked: bool = False
    block_reason: str = ""
    replicas_before: int
    replicas_after: int


@app.post("/api/decide")
async def decide(request: DecideRequest):
    if request.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    return state["engine"].decide(request.mode, ClusterMetrics(**request.metrics.model_dump()))


@app.post("/api/report")
async def report(request: ReportRequest):
    try:
        events = state["engine"].report(**request.model_dump())
    except (LookupError, ValueError) as e:
        raise HTTPException(409, str(e))
    await broadcaster.send({"type": "step", "status": state["engine"].snapshot(), **events})
    return {"status": "recorded"}


# ---------------------------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    engine = state["engine"]
    await ws.send_json({
        "type": "snapshot", "status": engine.snapshot(), "history": engine.recent_history(720),
        "audit": engine.recent_audit(100), "learning": engine.online_learning_curve(),
        "config": state["config"], "load": state["load"],
    })
    broadcaster.clients.add(ws)
    try:
        while True:
            await ws.receive_text()   # keep the connection open; the client sends nothing
    except WebSocketDisconnect:
        broadcaster.clients.discard(ws)


@app.get("/api/status")
def get_status():
    return state["engine"].snapshot()


@app.get("/api/history")
def get_history(limit: int = 720):
    return state["engine"].recent_history(limit)


@app.get("/api/audit")
def get_audit(limit: int = 100):
    return state["engine"].recent_audit(limit)


@app.get("/api/online-training")
def get_online_training():
    return state["engine"].online_learning_curve()


@app.get("/api/config")
def get_config():
    return state["config"]


@app.put("/api/config")
def update_config(changes: dict):
    """Partial update: {"section": {"key": value}}. Only existing numeric keys can change here."""
    config = copy.deepcopy(state["config"])
    for section, values in changes.items():
        if section not in config or not isinstance(values, dict):
            raise HTTPException(400, f"unknown section {section}")
        for key, value in values.items():
            if key not in config[section]:
                raise HTTPException(400, f"unknown key {section}.{key}")
            current = config[section][key]
            if isinstance(current, (int, float)) and not isinstance(value, (int, float)):
                raise HTTPException(400, f"{section}.{key} must be a number")
            config[section][key] = value
    validate_config(config)
    state["store"].set(CONFIG_PATH, config)
    return config


def validate_config(config: dict):
    limits = config["guardrails"]
    if not 1 <= limits["min_replicas"] <= limits["max_replicas"]:
        raise HTTPException(400, "guardrails: need 1 <= min_replicas <= max_replicas")
    for section, key in (("dqn", "gamma"), ("online", "active_epsilon")):
        if not 0 <= config[section][key] <= 1:
            raise HTTPException(400, f"{section}.{key} must be between 0 and 1")


class ModeRequest(BaseModel):
    mode: str


@app.post("/api/mode")
def set_mode(request: ModeRequest):
    if request.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    config = copy.deepcopy(state["config"])
    config["controller"]["mode"] = request.mode
    state["store"].set(CONFIG_PATH, config)
    return {"mode": request.mode}


class LoadRequest(BaseModel):
    profile: str                    # off | constant | spike | periodic
    base_qps: Optional[float] = None


@app.get("/api/load")
def get_load():
    return state["load"]


@app.post("/api/load")
def set_load(request: LoadRequest):
    if request.profile not in ("off", "constant", "spike", "periodic"):
        raise HTTPException(400, "profile must be off, constant, spike or periodic")
    load_cfg = state["config"]["load_profiles"]
    spec = {"profile": request.profile, "started_at": time.time()}
    if request.profile == "off":
        spec["base_qps"] = load_cfg["idle_qps"]   # background traffic, never fully silent
    else:
        spec.update({
            "base_qps": request.base_qps or load_cfg["default_base_qps"],
            "spike_multiplier": load_cfg["spike_multiplier"],
            "spike_after_seconds": load_cfg["spike_after_seconds"],
            "spike_duration_seconds": load_cfg["spike_duration_seconds"],
            "period_seconds": load_cfg["period_seconds"],
            "amplitude": load_cfg["amplitude"],
        })
    state["store"].set(LOAD_PATH, spec)
    return spec


# ---------------------------------------------------------------------------------------------
# Offline training results (files written by train_dqn.py, run_experiments.py, train_lstm.py)
# ---------------------------------------------------------------------------------------------

def _run_dirs():
    """Every training run folder, including the experiment runs. Runs from older versions of the
    code (without the current config format) are skipped."""
    found = {}
    for parent, prefix in ((RUNS_DIR, ""), (os.path.join(RUNS_DIR, "experiments"), "experiments/")):
        if not os.path.isdir(parent):
            continue
        for name in sorted(os.listdir(parent)):
            run_dir = os.path.join(parent, name)
            config_path = os.path.join(run_dir, "config.json")
            if not os.path.exists(os.path.join(run_dir, "metrics.csv")) or not os.path.exists(config_path):
                continue
            with open(config_path) as f:
                if "guardrails" not in json.load(f).get("config", {}):
                    continue
            found[prefix + name] = run_dir
    return found


@app.get("/api/runs")
def list_runs():
    runs = []
    for name, run_dir in _run_dirs().items():
        with open(os.path.join(run_dir, "config.json")) as f:
            args = json.load(f).get("args", {})
        runs.append({"name": name, "modified": os.path.getmtime(os.path.join(run_dir, "metrics.csv")),
                     "epsilon_mode": args.get("epsilon_mode"), "lstm": not args.get("no_lstm", False),
                     "has_evaluation": os.path.exists(os.path.join(run_dir, "evaluation.json"))})
    return sorted(runs, key=lambda r: r["modified"], reverse=True)


@app.get("/api/runs/{name:path}")
def get_run(name: str):
    run_dir = _run_dirs().get(name)
    if run_dir is None:
        raise HTTPException(404, "run not found")
    with open(os.path.join(run_dir, "metrics.csv"), newline="") as f:
        rows = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(f)]
    with open(os.path.join(run_dir, "config.json")) as f:
        run_config = json.load(f)
    evaluation = None
    if os.path.exists(os.path.join(run_dir, "evaluation.json")):
        with open(os.path.join(run_dir, "evaluation.json")) as f:
            evaluation = json.load(f)
    return {"name": name, "args": run_config.get("args"), "dqn": run_config["config"]["dqn"],
            "episodes": rows, "evaluation": evaluation}


@app.get("/api/experiments")
def get_experiments():
    folder = os.path.join(RUNS_DIR, "experiments")
    summary_path = os.path.join(folder, "summary.csv")
    if not os.path.exists(summary_path):
        return {"summary": [], "threshold": None}
    with open(summary_path, newline="") as f:
        summary = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(f)]
    threshold = None
    if os.path.exists(os.path.join(folder, "threshold_baseline.json")):
        with open(os.path.join(folder, "threshold_baseline.json")) as f:
            threshold = json.load(f)
    return {"summary": summary, "threshold": threshold}


@app.get("/api/models")
def get_models():
    info = {"dqn": None, "lstm": None}
    info_path = os.path.join(MODELS_DIR, "model_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info["dqn"] = json.load(f)
    lstm_path = os.path.join(MODELS_DIR, "lstm.pt")
    if os.path.exists(lstm_path):
        from predictor.lstm_forecaster import load_forecaster
        _, meta = load_forecaster(lstm_path)
        info["lstm"] = meta
    return info


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# The built dashboard (npm run build), so everything can be served from one port.
if os.path.isdir(DASHBOARD_BUILD):
    app.mount("/", StaticFiles(directory=DASHBOARD_BUILD, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
