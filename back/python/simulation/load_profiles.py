"""Load profiles: how much traffic (QPS) the service should receive at each step.

The three profiles match the dashboard's Simulation Panel:
- constant: steady traffic around a base level
- spike:    steady traffic, then a sudden jump that lasts a while, then back down
- periodic: a smooth wave, like daily traffic

A profile only gives the *target* QPS. The environment follows it with momentum and noise,
so the actual QPS changes gradually (see MockKubernetesEnv._advance_load).
"""
import math
import random
from dataclasses import dataclass

PROFILE_KINDS = ["constant", "spike", "periodic"]


@dataclass
class LoadProfile:
    kind: str
    base_qps: float
    spike_multiplier: float = 1.0
    spike_start: int = 0
    spike_steps: int = 0
    amplitude: float = 0.0
    period_steps: int = 1
    phase: float = 0.0

    def target_qps(self, step: int) -> float:
        if self.kind == "spike":
            in_spike = self.spike_start <= step < self.spike_start + self.spike_steps
            return self.base_qps * (self.spike_multiplier if in_spike else 1.0)
        if self.kind == "periodic":
            wave = math.sin(2 * math.pi * step / self.period_steps + self.phase)
            return self.base_qps * (1 + self.amplitude * wave)
        return self.base_qps


def make_profile(kind: str, rng: random.Random, env_cfg: dict) -> LoadProfile:
    """Creates a profile of the given kind with randomized parameters."""
    base_qps = rng.uniform(env_cfg["base_qps_min"], env_cfg["base_qps_max"])
    episode_steps = env_cfg["episode_steps"]

    if kind == "constant":
        return LoadProfile(kind, base_qps)

    if kind == "spike":
        return LoadProfile(
            kind, base_qps,
            spike_multiplier=env_cfg["spike_profile_multiplier"],
            spike_start=rng.randint(episode_steps // 5, episode_steps // 2),
            spike_steps=rng.randint(env_cfg["spike_steps_min"], env_cfg["spike_steps_max"]),
        )

    if kind == "periodic":
        return LoadProfile(
            kind, base_qps,
            amplitude=env_cfg["periodic_amplitude"],
            period_steps=env_cfg["periodic_period_steps"],
            phase=rng.uniform(0, 2 * math.pi),
        )

    raise ValueError(f"Unknown load profile: {kind}")


def profile_from_load_spec(spec: dict, load_cfg: dict, step_seconds: float) -> LoadProfile:
    """Builds a profile from the /autoscaler/load ZNode (written by the dashboard).

    The Go load generator reads the same ZNode and generates real traffic with the same shape;
    the simulator converts its seconds into control steps.
    """
    kind = spec.get("profile", "off")
    base = spec.get("base_qps", load_cfg["default_base_qps"])
    if kind == "off":
        return LoadProfile("constant", spec.get("base_qps", load_cfg["idle_qps"]))
    if kind == "constant":
        return LoadProfile("constant", base)
    if kind == "spike":
        return LoadProfile(
            "spike", base,
            spike_multiplier=spec.get("spike_multiplier", load_cfg["spike_multiplier"]),
            spike_start=round(spec.get("spike_after_seconds", load_cfg["spike_after_seconds"]) / step_seconds),
            spike_steps=round(spec.get("spike_duration_seconds", load_cfg["spike_duration_seconds"]) / step_seconds),
        )
    if kind == "periodic":
        return LoadProfile(
            "periodic", base,
            amplitude=spec.get("amplitude", load_cfg["amplitude"]),
            period_steps=max(1, round(spec.get("period_seconds", load_cfg["period_seconds"]) / step_seconds)),
        )
    raise ValueError(f"Unknown load profile: {kind}")
