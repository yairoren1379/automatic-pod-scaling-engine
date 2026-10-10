"""A causal mock of a Kubernetes Deployment, used to pre-train the agent.

Every metric is caused by something, so the networks can learn real cause and effect:

    QPS (load profile + momentum)
      -> load per pod ("utilization", 1.0 = pods at full capacity)
          -> CPU            rises linearly with utilization, capped at 100%
          -> RAM            buffers and cache take memory, released gradually
              -> latency    memory term ~ e^(k * (RAM - offset)), grows with memory pressure
              -> OOM crash  when RAM reaches the limit
          -> latency        queueing term: exponential near the CPU limit
          -> error rate     near saturation, requests beyond the in-flight cap are rejected
          -> crashes        CPU-starved pods fail their liveness probe and restart

The numbers are calibrated against measurements of the real target app on Minikube.

The agent changes the number of pods, and every change has a real cost:
- Scale up: a new pod serves traffic only after it starts, and then runs with cold caches at
  reduced capacity for a while. Reacting late is expensive, so forecasting load pays off.
- Scale down: the removed pod's in-flight requests fail (connection resets), and under RAM
  pressure its load lands on the remaining pods: a latency and error spike.
Because both costs show up in the metrics, the reward penalizes flapping (scale up, down, up...)
without ever looking at the action itself.

The environment also produces extreme events: random pod crashes and sharp QPS spikes.

The interface follows the Gymnasium convention, except that observations are raw
ClusterMetrics: the Predictor turns them into the DQN state, exactly as on the real cluster.
    reset() -> metrics
    step(action) -> (metrics, reward, terminated, truncated, info)
"""
import math
import random
from dataclasses import asdict

from core.actions import SCALE_UP, SCALE_DOWN
from core.reward import compute_reward
from core.state import ClusterMetrics
from simulation.load_profiles import PROFILE_KINDS, LoadProfile, make_profile

# Utilization is capped so that a deployment with no ready pods doesn't divide by zero.
MAX_UTILIZATION = 3.0


class MockKubernetesEnv:

    def __init__(self, config: dict, seed: int = None, profile_kind: str = None):
        """profile_kind fixes the load profile; None picks a random one every episode."""
        self.config = config
        self.env_cfg = config["mock_env"]
        self.min_replicas = config["guardrails"]["min_replicas"]
        self.max_replicas = config["guardrails"]["max_replicas"]
        self.scale_up_step = config["guardrails"]["scale_up_step"]
        self.scale_down_step = config["guardrails"]["scale_down_step"]
        # One step is one control loop, so the stabilization window in seconds becomes steps.
        self.stabilization_steps = round(config["guardrails"]["scale_down_stabilization_seconds"]
                                         / config["controller"]["loop_delay_seconds"])
        self.episode_steps = self.env_cfg["episode_steps"]
        # The most traffic we ever generate. Loads that even max_replicas can't serve would make
        # failures that are not the agent's fault, which only adds noise to learning.
        self.max_qps = (self.env_cfg["max_load_fraction"] * self.max_replicas
                        * self.env_cfg["qps_capacity_per_pod"])
        self.profile_kind = profile_kind
        self.rng = random.Random(seed)

    # ---------------------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------------------

    def reset(self) -> ClusterMetrics:
        kind = self.profile_kind or self.rng.choice(PROFILE_KINDS)
        self.profile = make_profile(kind, self.rng, self.env_cfg)
        self.profile_start_step = 0
        self.step_count = 0
        self.steps_since_scale_up = self.stabilization_steps   # no recent scale-up

        self.qps = min(self.profile.target_qps(0), self.max_qps)
        self.qps_trend = 0.0
        self.random_spike_steps_left = 0
        self.random_spike_multiplier = 1.0

        # Start near a sensible replica count (~60% utilization), give or take a few pods,
        # so the agent sees both over- and under-provisioned starting points.
        capacity = self.env_cfg["qps_capacity_per_pod"]
        sensible = math.ceil(self.qps / (capacity * 0.6))
        replicas = self.rng.randint(sensible - 3, sensible + 3)
        self.ready_pods = max(self.min_replicas, min(self.max_replicas, replicas))
        self.starting_pods = []    # steps left until each new pod is ready
        self.warming_pods = []     # steps left until each ready-but-cold pod is warm
        self.recovering_pods = []  # steps left until each crashed pod is back

        # Begin with memory already at its steady state for the current load.
        self.ram = self._ram_target(self._utilization())
        self.metrics = self._measure(scale_down_shock=0.0, termination_errors=0.0,
                                     allow_random_events=False)
        return self.metrics

    def step(self, action: int):
        previous = self.metrics

        self.steps_since_scale_up += 1   # one more control loop since the last scale-up
        block_reason = self.block_reason(action)
        blocked = bool(block_reason)
        scale_down_shock, termination_errors = 0.0, 0.0
        if not blocked:
            scale_down_shock, termination_errors = self._apply_action(action)
            if action == SCALE_UP:
                self.steps_since_scale_up = 0

        self.step_count += 1
        self._advance_load()
        self._advance_pods()
        self.metrics = self._measure(scale_down_shock, termination_errors, allow_random_events=True)

        reward = compute_reward(previous, self.metrics, blocked, self.config)

        # Autoscaling never "ends", so no state is terminal. The episode is only cut after a
        # fixed number of steps (truncated): the agent should still value the future there.
        terminated = False
        truncated = self.step_count >= self.episode_steps

        info = {
            "metrics": asdict(self.metrics),
            "reward": asdict(reward),
            "blocked": blocked,
            "block_reason": block_reason,
            "profile": self.profile.kind,
            "ready_pods": self.ready_pods,
        }
        return self.metrics, reward.total, terminated, truncated, info

    def set_profile(self, profile: LoadProfile):
        """Switches the load profile mid-episode (used by the interactive simulator)."""
        self.profile = profile
        self.profile_start_step = self.step_count
        self.random_spike_steps_left = 0

    @property
    def replicas(self) -> int:
        """The Deployment's desired replica count: every pod, ready or not."""
        return self.ready_pods + len(self.starting_pods) + len(self.recovering_pods)

    # ---------------------------------------------------------------------------------------
    # Actions and guardrails
    # ---------------------------------------------------------------------------------------

    def block_reason(self, action: int) -> str:
        """Mirror of the Go guardrails (back/go/controller/guardrails.go). Returns why the action
        would be blocked, or "" if it is allowed."""
        if action == SCALE_UP and self.replicas + self.scale_up_step > self.max_replicas:
            return "max_replicas"
        if action == SCALE_DOWN:
            if self.replicas - self.scale_down_step < self.min_replicas:
                return "min_replicas"
            if self.steps_since_scale_up < self.stabilization_steps:
                return "stabilization_window"
        return ""

    def _apply_action(self, action: int):
        """Changes the pod set. Returns (scale-down shock 0..1, error % from terminated pods)."""
        if action == SCALE_UP:
            for _ in range(self.scale_up_step):
                self.starting_pods.append(self.env_cfg["pod_startup_steps"])
            return 0.0, 0.0

        if action == SCALE_DOWN:
            shock, termination_errors = 0.0, 0.0
            for _ in range(self.scale_down_step):
                # Like Kubernetes, delete pods that aren't serving traffic first, then the
                # ready pod that has been ready for the shortest time.
                if self.starting_pods:
                    self.starting_pods.pop()
                elif self.recovering_pods:
                    self.recovering_pods.pop()
                else:
                    traffic_share = 1.0 / self.ready_pods
                    termination_errors += (100 * self.env_cfg["termination_failed_fraction"]
                                           * traffic_share)
                    shock = max(shock, self._memory_pressure())
                    self.ready_pods -= 1
                    if self.warming_pods:
                        self.warming_pods.pop()
            return shock, termination_errors

        return 0.0, 0.0

    def _memory_pressure(self) -> float:
        """0 below the pressure threshold, rising to 1 at 100% RAM."""
        threshold = self.env_cfg["pressure_ram_percent"]
        return max(0.0, (self.ram - threshold) / (100 - threshold))

    # ---------------------------------------------------------------------------------------
    # Time passing
    # ---------------------------------------------------------------------------------------

    def _advance_load(self):
        """Moves QPS toward the profile target with momentum, so it forms learnable trends."""
        cfg = self.env_cfg
        target = self.profile.target_qps(self.step_count - self.profile_start_step)

        # Extreme event: a sharp spike jumps immediately, without momentum.
        if self.random_spike_steps_left > 0:
            self.random_spike_steps_left -= 1
        elif self.rng.random() < cfg["random_spike_probability"]:
            self.random_spike_multiplier = self.rng.uniform(
                cfg["random_spike_multiplier_min"], cfg["random_spike_multiplier_max"])
            self.random_spike_steps_left = self.rng.randint(
                cfg["random_spike_steps_min"], cfg["random_spike_steps_max"])
            self.qps = min(self.qps * self.random_spike_multiplier, self.max_qps)
        if self.random_spike_steps_left > 0:
            target *= self.random_spike_multiplier
        target = min(target, self.max_qps)

        # The trend is a slowly drifting push up or down: it keeps its direction for a while
        # (momentum) and only changes through small random nudges.
        self.qps_trend = (cfg["qps_trend_momentum"] * self.qps_trend
                          + self.rng.gauss(0, cfg["qps_noise_std"]) * target)
        self.qps += cfg["qps_follow_rate"] * (target - self.qps) + self.qps_trend
        self.qps = max(0.0, min(self.qps, self.max_qps))

    def _advance_pods(self):
        """Pods move through their lifecycle: starting -> warming (cold caches) -> warm.
        Crashed pods come back after their recovery time, also cold."""
        cfg = self.env_cfg
        self.warming_pods = [steps - 1 for steps in self.warming_pods if steps > 1]

        self.starting_pods = [steps - 1 for steps in self.starting_pods]
        self.recovering_pods = [steps - 1 for steps in self.recovering_pods]
        now_ready = (sum(1 for steps in self.starting_pods if steps <= 0)
                     + sum(1 for steps in self.recovering_pods if steps <= 0))
        self.starting_pods = [steps for steps in self.starting_pods if steps > 0]
        self.recovering_pods = [steps for steps in self.recovering_pods if steps > 0]

        self.ready_pods += now_ready
        self.warming_pods += [cfg["warmup_steps"]] * now_ready

    def _crash_one_pod(self):
        self.ready_pods -= 1
        if self.warming_pods:
            self.warming_pods.pop()
        self.recovering_pods.append(self.env_cfg["crash_recovery_steps"])

    # ---------------------------------------------------------------------------------------
    # Measuring: from load to metrics
    # ---------------------------------------------------------------------------------------

    def _utilization(self) -> float:
        cfg = self.env_cfg
        warm_pods = self.ready_pods - len(self.warming_pods)
        capacity = cfg["qps_capacity_per_pod"] * (
            warm_pods + cfg["warmup_capacity_fraction"] * len(self.warming_pods))
        if capacity <= 0:
            return MAX_UTILIZATION
        return min(self.qps / capacity, MAX_UTILIZATION)

    def _ram_target(self, utilization: float) -> float:
        """Memory a pod settles at under this load. Keeps growing a little past full capacity,
        as queued requests pile up, until the in-flight request cap stops it."""
        cfg = self.env_cfg
        return cfg["base_ram_percent"] + cfg["ram_per_util_percent"] * min(utilization, cfg["ram_util_cap"])

    def _latency(self, utilization: float) -> float:
        """Service time + memory pressure term + CPU queueing term.

        The memory term is the e^(k * (RAM - offset)) curve from the design. The queueing term
        was added after measuring the real cluster: latency stays flat until the CPU limit is
        nearly reached, then requests queue and it grows exponentially (no hard step).
        """
        cfg = self.env_cfg
        memory_term = cfg["memory_latency_ms"] * math.exp(
            cfg["latency_ram_k"] * (self.ram / 100 - cfg["latency_ram_offset"]))
        queue_term = cfg["queue_latency_ms"] * math.exp(
            cfg["queue_latency_k"] * (min(utilization, 1.2) - cfg["queue_latency_knee"]))
        latency = (cfg["service_latency_ms"] + memory_term + queue_term) * (1 + self.rng.gauss(0, cfg["latency_noise"]))
        return min(max(latency, 1.0), cfg["max_latency_ms"])

    def _measure(self, scale_down_shock: float, termination_errors: float,
                 allow_random_events: bool) -> ClusterMetrics:
        cfg = self.env_cfg
        utilization = self._utilization()

        # Memory moves toward its target gradually (inertia), plus a bump if a pod was removed
        # under pressure and its in-flight requests landed on the remaining pods.
        self.ram = (cfg["ram_inertia"] * self.ram
                    + (1 - cfg["ram_inertia"]) * self._ram_target(utilization)
                    + cfg["scale_down_shock_ram_percent"] * scale_down_shock)
        self.ram = min(self.ram, 100.0)

        crashed = False
        if allow_random_events and self.ready_pods > 0:
            # A CPU-starved pod fails its liveness probe and is restarted.
            overload_crash = min(cfg["overload_crash_max"],
                                 max(0.0, utilization - cfg["overload_crash_start"]) * cfg["overload_crash_slope"])
            if self.ram >= cfg["oom_ram_percent"]:
                self._crash_one_pod()   # OOM kill: the remaining pods get even more load
                crashed = True
            elif self.rng.random() < overload_crash + cfg["random_crash_probability"]:
                self._crash_one_pod()
                crashed = True

        cpu = min(100.0, cfg["idle_cpu_percent"] + (100 - cfg["idle_cpu_percent"]) * utilization)
        latency = self._latency(utilization)

        # Near saturation the queue fills up and requests beyond the in-flight cap are rejected.
        overload = max(0.0, utilization - cfg["overload_error_start"])
        error_rate = 100 * cfg["overload_error_slope"] * overload / utilization if utilization > 0 else 0.0
        error_rate += cfg["scale_down_shock_error_percent"] * scale_down_shock
        error_rate += termination_errors
        if crashed:
            error_rate += cfg["crash_error_spike_percent"]
        error_rate = min(error_rate, 100.0)

        return ClusterMetrics(
            cpu_percent=cpu,
            ram_percent=self.ram,
            latency_ms=latency,
            qps=self.qps,
            error_rate_percent=error_rate,
            replicas=self.replicas,
            pod_crashed=crashed,
        )
