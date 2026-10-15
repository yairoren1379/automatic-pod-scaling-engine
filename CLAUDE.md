# Autonomous Kubernetes Pod Scaling Engine

Final project for a practical engineering (Handasai) diploma at ORT Rehovot: a Deep RL controller
that replaces threshold-based autoscaling in Kubernetes. The student will present and defend this
project, so prefer clear, well-commented, explainable code over clever abstractions, and explain
design decisions when making them.

## Working with the user
- Talk to the user in Hebrew. Write code, identifiers, comments and commit messages in English.
- Follow the architecture below. Don't swap core technologies (Go controller, Python DRL, React,
  ZooKeeper, file-based persistence) without asking first. They are part of an approved proposal.
- The project is finished. On a machine other than the development PC (the 16GB defense laptop),
  follow `docs/LAPTOP_SETUP.md`: install, run and verify only. No retraining or redesign.

## The problem
Static rules ("scale up at 80% CPU") cause flapping, ignore latency and QPS, and react only after
load has already arrived. Goal: a learning controller that anticipates load (QPS is a leading
indicator), filters momentary noise, and balances availability, performance and resource cost.

## Architecture (microservices)

### 1. AI & Learning Engine (Python)
- Predictor: moving-average smoothing + hysteresis (dual thresholds) against flapping, then an
  LSTM that forecasts CPU, RAM, Latency and QPS from recent history.
- DQN agent: state = current metrics + LSTM forecasts + replica count. Actions: scale up,
  scale down, hold. Epsilon-greedy with decay `eps = max(eps_min, eps_0 / (k*t + 1))`, plus a
  constant-epsilon run for comparison. Experience replay buffer.
- Learning: TD target `r + gamma * max_a' Q(s', a')`, loss on the gap, backpropagation.
- Exposes a REST API (FastAPI) to the Go controller: receives state + reward, returns an action.

### 2. Cluster Controller & Guardrails (Go, client-go)
- Metrics Collector: pulls CPU, RAM, Latency and QPS from the cluster.
- Hard Guardrails: validates every action before execution (min/max replicas, illegal commands).
  Blocked actions are never executed; the agent receives a virtual penalty instead.
- Action Executor: scales Deployments on Minikube.
- Modes: `baseline` (deterministic thresholds, for comparison), `shadow` (DQN decides and learns,
  nothing is applied), `active` (DQN decisions applied through the guardrails).

### 3. Frontend Dashboard (React SPA, HTML/CSS)
- Dashboard: live CPU/RAM/Latency/QPS, active replicas, DQN decisions vs guardrail blocks.
- Training Analytics: loss, cumulative reward, epsilon decay.
- Simulation Panel: trigger load profiles (constant, sudden spike, periodic).
- API Client: REST for commands and config, WebSocket for live streaming.

## Communication
- REST/HTTP with JSON between all services and to the Kubernetes API.
- WebSocket for live data to the dashboard (no polling).
- JSON is the single data format across Go, Python and JavaScript.

## Persistence: DB-less (do not add a relational database)
- Apache ZooKeeper (ZNodes): hyperparameters (epsilon schedule, gamma, learning rate), guardrail
  limits, environment settings. Services watch ZNodes and apply changes on the fly, no restarts.
- Local file system:
  - Replay buffer: kept in RAM, periodically saved (pickle/JSON).
  - Time-series metrics: CSV, for LSTM training and dashboard history.
  - Model checkpoints: DQN + LSTM weights, for recovery and sim-to-real transfer.
  - Safety & audit log: every agent decision vs every guardrail intervention.

## Mock environment (`mock_env.py`): all training starts here
Causal, not purely random, so the networks can learn real gradients:
- Momentum: CPU and QPS change gradually, forming trends the LSTM can predict.
- Latency rises exponentially with memory pressure, `latency ~ e^(k * (RAM - offset))`
  (page faults / thrashing). No hard step at 80%.
- Scale up: lowers QPS per replica, balances CPU, improves latency, raises resource cost.
- Aggressive scale down under RAM pressure: latency spike and error-rate jump.
- Must also produce extreme events: service crash, sharp QPS spike.

## Reward
- Severe failure (pod crash, NotReady, extreme latency, error-rate spike): r = -100.
- Plus: proportional to latency improvement since the previous step.
- Minus: proportional to error-rate increase.
- Minus: resource waste (many replicas while CPU stays low).
- Minus: virtual penalty for guardrail-blocked actions.
- Reward is computed from the resulting state, never from the action label: scale up earns
  reward only when it actually helped, and is penalized as wasteful when load is low.

## Training pipeline (sim-to-real)
1. Pre-train the LSTM and DQN in the mock env (thousands of episodes in minutes).
2. Minikube baseline: run the threshold controller under all load profiles, record results.
3. Shadow mode on the real cluster until the learning curve is stable against real noise.
4. Active mode: lower learning rate and epsilon, fine-tuning only.
Experiments: constant vs decaying epsilon; a grid of learning rate x gamma.

## Environment
- Local development on Minikube. The development machine has 32GB RAM and 20 logical CPUs
  (upgraded from 16GB on 2026-10-14); Docker Desktop gets ~15.5GB. Minikube runs with 6GB / 6
  CPUs. Target pods are limited to 200m CPU / 256Mi, max 15 replicas. README real-cluster results:
  run 1 on 6 CPUs (160 real transitions), run 2 on 8 CPUs (639, after a 4-hour shadow stage).
- Windows host with Smart App Control ON: it blocks the locally compiled Go controller binary
  (`go run ./controller` fails). The controller therefore runs inside the cluster
  (`back/k8s/controller.yaml`, least-privilege RBAC) and reaches the API on the host through
  `host.minikube.internal:8000`. Go programs are built inside Docker; images are loaded with
  `minikube image load` (the cluster may use the containerd runtime).
  Smart App Control also judges each DLL by cloud reputation: torch 2.14.1's `shm.dll` was
  blocked (WinError 4551, `import torch` fails), so torch is pinned to 2.13.0. If a torch
  upgrade breaks the import, check the CodeIntegrity event log (events 3033/3077) for the file.
- Minikube's metrics-server addon defaults to a 60s resolution, and every `minikube start`
  resets it. `scripts/patch-metrics-server.ps1` sets 15s so the 30s control loop never sees
  stale CPU/RAM; both setup-minikube.ps1 and start-system.ps1 run it.
- ZooKeeper keeps its data on a PVC (`back/k8s/zookeeper.yaml`, Recreate strategy), so the
  ZNodes survive reboots. The controller waits for the config ZNode instead of exiting.
- Python venv at `back/python/.venv` (PyTorch CPU). Go module `autoscaler` in `back/go`
  (controller, targetapp, loadgen). React (CRA) in `front`.
- `README.md` documents the architecture, every command and the API; keep it in sync.
  `docs/calibration.md` records the real-cluster measurements the mock env is calibrated to.

## Commands
- Python tests: `cd back/python; .venv/Scripts/python -m pytest tests`
- Go tests: `cd back/go; go test ./...`; dashboard: `cd front; npm test`
- Training: `train_lstm.py` -> `train_dqn.py` / `run_experiments.py` -> `evaluate_dqn.py` ->
  `promote_model.py` (copies a run to `models/`, which the API server loads).
- Run without a cluster: `scripts/start-simulation.ps1` (API with ZK_DISABLED=1 +
  `simulate_controller.py` + dashboard). With Minikube: `scripts/setup-minikube.ps1` (also after
  Go changes: rebuilds images and restarts pods), then `scripts/start-system.ps1`.
- Real-cluster pipeline stages: `run_cluster_stages.py --stage baseline|shadow|active`, then
  `--compare <run dirs>`. Results in `runs/cluster/`.
- Background commands in this environment are killed after 10 minutes: launch long training runs
  as detached processes (PowerShell `Start-Process`).

## Settled decisions
- DL framework: PyTorch. Checkpoints are `.pt` files.
- Actions: scale up, scale down, hold. Restart is dropped.
- Error rate is part of the state: the metric vector is CPU, RAM, Latency, QPS, Error rate.
  The DQN state is 11 values: smoothed metrics (incl. replicas) + 4 LSTM forecasts + hysteresis
  load regime.
- The DQN uses a target network (synced every N steps) and a Double DQN target for training
  stability. Additions to the proposal; justify them in the project book.
- Model selection: training validates the greedy policy every 50 episodes on fixed seeds
  (5000+); `dqn.pt` is the best, `dqn_last.pt` the last. Final evaluation uses seeds 10000+.
- Reward adds a "latency above SLO" term to the proposal's terms (otherwise a constantly high
  latency is never penalized). Its weight is configurable; 0 restores the proposal exactly.
- Anti-flapping is a guardrail: no scale-down within `scale_down_stabilization_seconds` (150) of
  a scale-up, like the HPA. Blocked actions get the virtual penalty. Scale-up is never delayed.
  The mock env also models churn costs causally (pod warm-up, failed requests on termination).
- Latency/QPS source: the target app (`back/go/targetapp`) exposes JSON counters at `/metrics`;
  the Go controller reads them per pod through the Kubernetes API proxy. No Prometheus.
- The mock env is calibrated to the real target app (docs/calibration.md). Latency = service time
  + the design's memory term e^(k*(RAM-offset)) + a CPU-queueing term that grows exponentially
  near the CPU limit (the measured cliff). Errors start at ~85% utilization; CPU-starved pods
  restart (liveness probe). Re-run the calibration if the target app or its limits change.
- Shadow mode: the threshold baseline drives the cluster and the DQN learns off-policy from the
  executed actions. Active mode falls back to the baseline if the engine is unreachable.
- Only real cause and effect is learned. `run_cluster_stages.py` switches to baseline mode
  (no learning) before every reset, so the transition spanning a reset is never learned. The
  controller skips a loop when a reading is missing (metrics-server not ready, first reading of
  the app counters, no ready pods) instead of reporting zeros. Online learning persists its
  learning curve in `data/online_learning.csv` and resumes `online_steps` from the checkpoint.
- The model fine-tuned on the real cluster is shipped in git as `back/python/models/online_dqn.pt`
  + `online_replay.pkl` (copied from data/ after run 2, 660 online steps). The engine starts from
  data/ first, then that, then the simulation-only `models/dqn.pt`. Re-copy after more learning.
- Without ZooKeeper, the offline scripts and the API (ZK_DISABLED=1) use DEFAULT_CONFIG.
- The tabular Q-learning + bandit prototype lives in `back/python/legacy/`, not used.

## Open decisions
Raise these when they become relevant instead of silently picking one. Once settled, update this file.
- The proposal's timeline still lists a Multi-Armed Bandit stage and plain Q-Learning, while the
  body describes Safe RL (guardrails + reward penalties) with DQN. The code now follows the body;
  the student should align the proposal/project book (the prototype is kept in `legacy/`).
