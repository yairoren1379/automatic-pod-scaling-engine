"""The reward function.

The reward is computed from the state the cluster ended up in, never from the action label.
Scaling up is not "good" by itself: it earns reward only if latency actually improved, and it
is penalized as waste if the load did not need the extra replicas.

The same function is used in the mock environment and on the real cluster.
"""
from dataclasses import dataclass

from core.state import ClusterMetrics


@dataclass
class RewardBreakdown:
    """The total reward and the term each part contributed (shown on the dashboard)."""
    total: float
    severe_failure: bool
    latency_improvement: float = 0.0
    slo_violation: float = 0.0
    error_increase: float = 0.0
    waste: float = 0.0
    blocked_action: float = 0.0


def is_severe_failure(metrics: ClusterMetrics, reward_cfg: dict) -> bool:
    return (
        metrics.pod_crashed
        or metrics.latency_ms >= reward_cfg["extreme_latency_ms"]
        or metrics.error_rate_percent >= reward_cfg["error_spike_percent"]
    )


def compute_reward(previous: ClusterMetrics, current: ClusterMetrics, action_blocked: bool,
                   config: dict) -> RewardBreakdown:
    reward_cfg = config["reward"]
    limits = config["guardrails"]

    # A guardrail block is penalized on top of everything else: the action was illegal.
    blocked_penalty = reward_cfg["blocked_action_penalty"] if action_blocked else 0.0

    # A severe failure overrides the shaping terms: nothing else matters if the service is down.
    if is_severe_failure(current, reward_cfg):
        return RewardBreakdown(
            total=reward_cfg["severe_failure"] + blocked_penalty,
            severe_failure=True,
            blocked_action=blocked_penalty,
        )

    slo = reward_cfg["latency_slo_ms"]

    # Plus: latency got better since the previous step (minus if it got worse).
    latency_improvement = reward_cfg["latency_improvement_weight"] * (
        (previous.latency_ms - current.latency_ms) / slo)

    # Minus: latency is above the SLO. Without this term the agent would only be paid for
    # changes in latency, so it could sit at a constantly high latency for free.
    slo_violation = -reward_cfg["slo_violation_weight"] * max(0.0, current.latency_ms - slo) / slo

    # Minus: new errors appeared (per percentage point).
    error_increase = -reward_cfg["error_increase_weight"] * max(
        0.0, current.error_rate_percent - previous.error_rate_percent)

    # Minus: many replicas while CPU stays low. Grows both with how idle the pods are and with
    # how many replicas are running above the minimum.
    waste_cpu = reward_cfg["waste_cpu_percent"]
    idle_fraction = max(0.0, waste_cpu - current.cpu_percent) / waste_cpu
    # max(1, ...): with min == max the agent has no choice, so there is no waste to blame on it.
    extra_replicas_fraction = (max(0, current.replicas - limits["min_replicas"])
                               / max(1, limits["max_replicas"] - limits["min_replicas"]))
    waste = -reward_cfg["waste_weight"] * idle_fraction * extra_replicas_fraction

    total = latency_improvement + slo_violation + error_increase + waste + blocked_penalty
    return RewardBreakdown(
        total=total,
        severe_failure=False,
        latency_improvement=latency_improvement,
        slo_violation=slo_violation,
        error_increase=error_increase,
        waste=waste,
        blocked_action=blocked_penalty,
    )
