"""Writes the default configuration to ZooKeeper.

Every tunable value of the system lives in DEFAULT_CONFIG. Run this script once after starting
ZooKeeper (and again after changing a default). At runtime the services watch the ZNodes, so a
value changed in ZooKeeper (for example from the dashboard) applies without a restart.

ZNodes:
    /autoscaler/config   this whole dictionary
    /autoscaler/load     the active load profile, written by the dashboard, read by the load generator
"""
import json
import os

from kazoo.client import KazooClient

ZK_HOSTS = os.environ.get("ZK_HOST", "127.0.0.1:2181")
CONFIG_PATH = "/autoscaler/config"
LOAD_PATH = "/autoscaler/load"

DEFAULT_LOAD = {"profile": "off", "base_qps": 20}   # idle background traffic

DEFAULT_CONFIG = {
    # Hard limits enforced by the Go guardrails (and mirrored by the mock environment).
    "guardrails": {
        "min_replicas": 1,
        "max_replicas": 15,
        "scale_up_step": 1,         # replicas added by one ScaleUp
        "scale_down_step": 1,       # replicas removed by one ScaleDown
        # Anti-flapping: no scale-down within this long after a scale-up (like the Kubernetes
        # HPA's scale-down stabilization window). Scaling up is never delayed.
        "scale_down_stabilization_seconds": 150
    },

    # The Go controller.
    "controller": {
        "mode": "shadow",           # baseline | shadow | active
        "loop_delay_seconds": 30,
        "target_namespace": "default",
        "target_deployment": "yair-api",
        "target_label": "app=yair-api",
        "app_metrics_port": "8080",
        "app_metrics_path": "/metrics"
    },

    # The deterministic threshold controller: the comparison baseline, and what drives the
    # cluster in baseline and shadow modes.
    "baseline": {
        "scale_up_cpu_percent": 80,
        "scale_down_cpu_percent": 30
    },

    # ---------------------------------------------------------------------------------------
    # Learning engine
    # ---------------------------------------------------------------------------------------

    "dqn": {
        "hidden_size": 128,             # neurons in each of the two hidden layers
        # Learning rate and gamma were chosen by the grid in run_experiments.py (best reward
        # and fewest failures, with the smallest spread between seeds).
        "learning_rate": 0.001,         # Adam step size
        "gamma": 0.95,                  # discount: looks ~20 steps (~10 minutes) ahead
        "batch_size": 64,               # transitions per gradient step
        "replay_capacity": 100000,      # transitions kept in the replay buffer (500 episodes)
        "min_replay_size": 1000,        # don't learn until the buffer has this many samples
        "target_sync_steps": 500,       # copy the online network into the target network
        "double_dqn": True,             # online net picks the next action, target net rates it
        "grad_clip_norm": 10.0,         # cap on the gradient norm, protects against -100 spikes
        "epsilon_start": 1.0,           # eps_0 in eps = max(eps_min, eps_0 / (k*t + 1))
        "epsilon_min": 0.05,
        "epsilon_decay_k": 0.0001,      # k: reaches eps_min after ~190k steps (~950 episodes)
        "epsilon_constant": 0.1,        # used by the constant-epsilon comparison run
        "episodes": 1500,
        "save_every_episodes": 100,     # checkpoint + replay buffer save interval
        "validate_every_episodes": 50,  # greedy test on fixed validation episodes; best is kept
        "validation_episodes_per_profile": 4
    },

    # Fine-tuning on the real cluster (shadow and active modes).
    "online": {
        "shadow_learning_rate": 0.0005,
        "active_learning_rate": 0.0001, # lower: active mode only fine-tunes
        "active_epsilon": 0.02,         # a little exploration, through the guardrails
        "batch_size": 32,
        "min_replay_size": 64,          # real transitions are slow to collect (one per loop)
        "updates_per_step": 4,          # gradient steps per control loop
        "save_every_steps": 20,
        "history_points": 720           # points kept in memory for the dashboard (6h at 30s)
    },

    "predictor": {
        "smoothing_window": 3,          # moving average over the last 3 readings
        # Hysteresis (dual thresholds) on smoothed CPU: the load regime switches to "high"
        # above 75% but only returns to "normal" below 60%, so it doesn't flicker at a boundary.
        "high_enter_cpu_percent": 75,
        "high_exit_cpu_percent": 60,
        "low_enter_cpu_percent": 25,
        "low_exit_cpu_percent": 35,
        # LSTM forecaster
        "sequence_length": 12,          # readings of history fed to the LSTM (6 minutes)
        "horizon_steps": 3,             # how far ahead it forecasts (1.5 minutes)
        "hidden_size": 32,
        "learning_rate": 0.002,
        "epochs": 15,
        "batch_size": 256,
        "training_episodes": 300        # mock episodes collected to train the LSTM
    },

    # How raw metrics are scaled into the [0, ~1] range the neural networks expect.
    # Changing these invalidates trained checkpoints.
    "state_normalization": {
        "qps_scale": 1500,              # max_replicas * per-pod capacity
        "latency_scale_ms": 1000,
        "error_scale_percent": 20
    },

    "reward": {
        "severe_failure": -100.0,       # pod crash, extreme latency or error-rate spike
        "extreme_latency_ms": 600,
        "error_spike_percent": 10,
        "latency_slo_ms": 200,          # latency target, used to scale the latency terms
        "latency_improvement_weight": 1.0,
        "slo_violation_weight": 1.0,
        "error_increase_weight": 0.5,   # per percentage point of new errors
        "waste_cpu_percent": 40,        # below this CPU, extra replicas count as waste
        "waste_weight": 2.0,
        "blocked_action_penalty": -5.0  # virtual penalty when a guardrail blocks the action
    },

    # Parameters of the load profiles the dashboard can trigger on the real cluster.
    "load_profiles": {
        "default_base_qps": 400,
        "idle_qps": 20,                 # background traffic when the load is "off"
        "spike_multiplier": 2.0,
        "spike_after_seconds": 120,
        "spike_duration_seconds": 600,
        "period_seconds": 1800,
        "amplitude": 0.5
    },

    # The mock environment, calibrated against measurements of the target app on Minikube
    # (docs/calibration.md): CPU, memory, latency and errors at 30-150 requests/s per pod.
    "mock_env": {
        "episode_steps": 200,               # one step = one control loop (~30s)

        # Capacity and resource model
        "qps_capacity_per_pod": 100,        # QPS one pod serves at 100% CPU
        "idle_cpu_percent": 15,             # runtime, GC and HTTP overhead
        "base_ram_percent": 15,             # memory of an idle pod
        "ram_per_util_percent": 65,         # extra memory at full load (buffers and cache)
        "ram_util_cap": 1.1,                # the in-flight request cap stops memory growing beyond this
        "ram_inertia": 0.5,                 # memory is released gradually, not instantly
        "oom_ram_percent": 98,              # above this the pod is OOM-killed
        "pressure_ram_percent": 60,         # above this a scale-down causes a shock

        # Latency = service time + memory term + CPU-queueing term
        "service_latency_ms": 22,           # the request's own work and downstream call
        "memory_latency_ms": 3,             # memory term: 3ms * e^(k * (RAM - offset)),
        "latency_ram_k": 6.0,               # growing exponentially with memory pressure
        "latency_ram_offset": 0.5,
        "queue_latency_ms": 1300,           # queueing term: 1300ms * e^(k * (util - knee)),
        "queue_latency_knee": 0.88,         # measured: ~2ms extra at 60% utilization,
        "queue_latency_k": 23.0,            # 1.3s at 88% (CPU throttling makes requests queue)
        "max_latency_ms": 1500,            # the in-flight cap bounds the queue
        "latency_noise": 0.03,

        # Errors and crashes under overload (measured: errors from ~85% utilization, and pods
        # restarted by failing liveness probes when CPU-starved)
        "overload_error_start": 0.85,
        "overload_error_slope": 1.5,
        "overload_crash_start": 0.95,
        "overload_crash_slope": 2.0,
        "overload_crash_max": 0.5,

        # Pod lifecycle
        "pod_startup_steps": 1,             # a new pod is ready by the next control loop
        "warmup_steps": 1,                  # then runs cold for one step...
        "warmup_capacity_fraction": 0.7,    # ...at reduced capacity
        "termination_failed_fraction": 0.15,# share of a removed pod's requests that fail
        "crash_recovery_steps": 3,          # a crashed pod is back after this many steps
        "random_crash_probability": 0.001,  # crash that is not caused by load
        "crash_error_spike_percent": 5,
        "scale_down_shock_ram_percent": 10,
        "scale_down_shock_error_percent": 8,

        # Traffic: momentum toward the profile target, plus a drifting trend
        "qps_follow_rate": 0.3,
        "qps_trend_momentum": 0.9,
        "qps_noise_std": 0.01,
        "max_load_fraction": 0.95,          # never ask for more than 95% of max_replicas capacity

        # Load profiles
        "base_qps_min": 150,
        "base_qps_max": 700,
        "spike_profile_multiplier": 1.8,
        "spike_steps_min": 20,
        "spike_steps_max": 60,
        "periodic_amplitude": 0.5,
        "periodic_period_steps": 60,

        # Random sharp QPS spikes (extreme events on top of any profile)
        "random_spike_probability": 0.01,
        "random_spike_multiplier_min": 1.5,
        "random_spike_multiplier_max": 2.2,
        "random_spike_steps_min": 5,
        "random_spike_steps_max": 15
    }
}


def setup_zookeeper_config():
    print(f"Connecting to ZooKeeper at {ZK_HOSTS}...")
    zk = KazooClient(hosts=ZK_HOSTS)
    zk.start(timeout=10)
    print("Connected!")

    for path, value in ((CONFIG_PATH, DEFAULT_CONFIG), (LOAD_PATH, DEFAULT_LOAD)):
        data = json.dumps(value, indent=4).encode("utf-8")
        if zk.exists(path):
            zk.set(path, data)
            print(f"UPDATED {path}")
        else:
            zk.create(path, data, makepath=True)
            print(f"CREATED {path}")

    zk.stop()


if __name__ == "__main__":
    setup_zookeeper_config()
