# Setting up the defense laptop

Instructions for Claude Code on the laptop the project is demonstrated on. The project itself is
finished: code, trained models and results are all in git. The job on the laptop is to install
the tools, run the system, and verify it works. **Do not retrain anything, change the design, or
"improve" code** unless the student asks; the README results come from this exact code and these
models. Talk to the student in Hebrew (see `CLAUDE.md`).

The laptop: 16 GB RAM (DDR4-3200), Intel i5-1235U (10 cores, 12 logical CPUs: 2 performance +
8 efficiency cores, 15 W). Windows. Minikube gets 6 GB / 6 CPUs (fixed in
`scripts/setup-minikube.ps1`), which leaves ~8 GB and 6 CPUs for Windows, VS Code and a browser.
The real-cluster run 1 used exactly that size without problems.

## 1. Install the tools (once)

Check what is already installed first (`docker --version`, `minikube version`, `kubectl version
--client`, `python --version`, `node --version`, `git --version`) and only install what is missing:

```powershell
winget install Docker.DockerDesktop      # then start Docker Desktop once and finish its setup
winget install Kubernetes.minikube Kubernetes.kubectl
winget install Python.Python.3.13        # 3.12-3.14 all work with torch 2.13.0
winget install OpenJS.NodeJS.LTS
```

Go is not needed: the Go programs are built inside Docker. (Only `go test` needs Go.)

Docker Desktop must have at least 7 GB of memory (by default it gets half the host's RAM, about
7.8 GB on this laptop). Check with `docker info --format "{{.MemTotal}}"`; the setup script
also checks it and explains what to do if it is too low.

## 2. Python and dashboard dependencies (once)

```powershell
cd back\python
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -c "import torch; print(torch.__version__)"   # must print 2.13.0+cpu
.venv\Scripts\python -m pytest tests -q                             # expect: 30 passed

cd ..\..\front
npm install
npm test -- --watchAll=false                                        # expect: 1 passed
```

Keep torch at **2.13.0**. On the development machine Windows Smart App Control blocked
torch 2.14.1's `shm.dll` (`OSError: [WinError 4551] An Application Control policy has blocked
this file`). If `import torch` fails like that here too: find the blocked file in Event Viewer →
Applications and Services Logs → Microsoft → Windows → CodeIntegrity → Operational (events 3033
/ 3077), and try torch 2.12.1. Don't turn Smart App Control off without asking the student
(it is a security setting, and on some Windows versions it can't be turned back on).

## 3. Create the cluster and start the system

Docker Desktop must be running.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-minikube.ps1   # ~5-10 min the first time
powershell -ExecutionPolicy Bypass -File scripts\start-system.ps1     # opens 4 windows
```

`setup-minikube.ps1` creates Minikube (6 GB / 6 CPUs), sets metrics-server to 15 s, builds the
three Go images and deploys everything. If a Minikube cluster already exists on the laptop with
another size, run `minikube delete` first (Minikube cannot resize a cluster).

`start-system.ps1` starts Minikube if it is stopped, re-applies the metrics-server setting, and
opens: the ZooKeeper port-forward (restarts itself), the API server on :8000, the controller's
log, and the dashboard on :3000 (`npm start`).

## 4. Verify (do all of these before the demo)

```powershell
kubectl get pods                       # zookeeper, yair-api, loadgen, autoscaler-controller: Running
kubectl -n kube-system get deployment metrics-server -o jsonpath="{.spec.template.spec.containers[0].args}"
                                       # must contain --metric-resolution=15s
kubectl logs deployment/autoscaler-controller --tail=5
                                       # a line every 30 s: "[mode] cpu=..% ... | dqn=<action> -> ..."
curl http://localhost:8000/api/status  # "connected": true, and model_source
                                       # "fine-tuned on the real cluster (...models\online_dqn.pt)"
```

`model_source` must say **fine-tuned on the real cluster**: that is the model from the README's
run 2 (`back/python/models/online_dqn.pt`, 660 real transitions). If it says "pre-trained in
simulation", the shipped model was not found. Check that `git pull` brought
`back/python/models/online_dqn.pt`. The API starts from `data/online_dqn.pt` if it exists, then
from `models/online_dqn.pt`, then from `models/dqn.pt`. A `data/` folder copied from another
machine would win.

Then open http://localhost:3000. Under *Simulation & control*, choose a load profile. Within a
minute the live dashboard should show traffic, replicas changing and DQN decisions.

## 5. During the demo

- **Mode.** In `shadow` or `baseline` the model cannot change behaviour. The dashboard still
  shows what the DQN proposes next to what was executed. `active` lets the DQN drive. It also
  keeps fine-tuning slowly (into `back/python/data/`, never into `models/`). That is the real
  system and is fine to show. To return to the shipped model afterwards, delete
  `back/python/data/online_dqn.pt` and `online_replay.pkl` and restart the API.
- **A good demo.** `active` mode with the **periodic** profile (where the DQN beats the
  thresholds most clearly), or the **spike** profile at 300 requests/s to show scaling up to 10+
  replicas. Each profile needs a few minutes to show anything; the control loop is 30 s.
- **The laptop.** Keep it plugged in (on battery Windows slows the i5 down). Close Discord, games
  and other heavy apps. Pause OneDrive sync if the project is inside OneDrive: it re-uploads the
  files the system writes every 30 s.
- **Fallback with no cluster.** If Docker/Minikube can't run on the laptop at all, use
  `scripts\start-simulation.ps1`: the same API, DQN, LSTM and dashboard against the simulated
  cluster. It needs only Python and Node. The real-cluster results are in the README.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `minikube start` fails: not enough memory | Docker Desktop has < 7 GB. Raise it in `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=8GB`), then `wsl --shutdown` and restart Docker Desktop |
| Controller log: `waiting for /autoscaler/config in ZooKeeper` | The API server isn't running yet. It creates the config on start, and the controller then continues by itself |
| Controller log: `collect: metrics-server ...` or `no request rates yet` | Normal for the first 1-2 loops after a start. The controller skips loops without valid readings |
| Controller log: `learning engine unreachable` | API on the host not running, or a firewall blocks port 8000 from the cluster (`host.minikube.internal:8000`). The controller falls back to the thresholds |
| CPU stays at 0% / stale | metrics-server reset to 60 s by `minikube start`. Run `scripts\patch-metrics-server.ps1` |
| Port 8000 or 3000 in use | Another copy is already running. Close the old windows, or stop the process |
| Go program blocked ("Application Control policy") | Expected with Smart App Control. Everything Go runs inside the cluster, so never run `go run` on the host |
| Dashboard on :8000 shows nothing | :8000 serves `front/build`, which only exists after `npm run build`. Use :3000 (started by `start-system.ps1`), or run `npm run build` once |

## 7. What not to do

- Don't retrain or re-promote models (`train_dqn.py`, `promote_model.py`, `run_experiments.py`):
  `models/dqn.pt` and `models/online_dqn.pt` are the ones the README's results describe.
- Don't commit `back/python/data/` or `back/python/runs/` (they are in `.gitignore` on purpose).
- Don't change the actions, reward, guardrails or the cluster size without the student.
- Commit or push only when the student asks.
