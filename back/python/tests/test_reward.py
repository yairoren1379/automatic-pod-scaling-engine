from dataclasses import replace

from core.reward import compute_reward
from core.state import ClusterMetrics


def metrics(**overrides):
    base = ClusterMetrics(cpu_percent=60, ram_percent=55, latency_ms=80, qps=500,
                          error_rate_percent=0, replicas=8)
    return replace(base, **overrides)


def test_severe_failure_overrides_everything(config):
    for failed in (metrics(pod_crashed=True), metrics(latency_ms=700), metrics(error_rate_percent=12)):
        reward = compute_reward(metrics(), failed, action_blocked=False, config=config)
        assert reward.severe_failure
        assert reward.total == config["reward"]["severe_failure"]


def test_latency_improvement_is_rewarded_and_degradation_penalized(config):
    better = compute_reward(metrics(latency_ms=150), metrics(latency_ms=100), False, config)
    worse = compute_reward(metrics(latency_ms=100), metrics(latency_ms=150), False, config)
    assert better.latency_improvement > 0 > worse.latency_improvement


def test_slo_violation_penalizes_constant_high_latency(config):
    reward = compute_reward(metrics(latency_ms=300), metrics(latency_ms=300), False, config)
    assert reward.latency_improvement == 0
    assert reward.slo_violation < 0


def test_only_error_increases_are_penalized(config):
    up = compute_reward(metrics(error_rate_percent=0), metrics(error_rate_percent=4), False, config)
    down = compute_reward(metrics(error_rate_percent=4), metrics(error_rate_percent=0), False, config)
    assert up.error_increase == -0.5 * 4
    assert down.error_increase == 0


def test_waste_grows_with_idle_replicas(config):
    few = compute_reward(metrics(), metrics(cpu_percent=10, replicas=3), False, config)
    many = compute_reward(metrics(), metrics(cpu_percent=10, replicas=12), False, config)
    busy = compute_reward(metrics(), metrics(cpu_percent=70, replicas=12), False, config)
    assert many.waste < few.waste < 0
    assert busy.waste == 0


def test_fixed_replica_count_does_not_crash(config):
    """min == max is a valid configuration (found on the real cluster: it divided by zero)."""
    config["guardrails"]["min_replicas"] = config["guardrails"]["max_replicas"] = 2
    reward = compute_reward(metrics(), metrics(cpu_percent=10, replicas=2), False, config)
    assert reward.waste == 0


def test_blocked_action_gets_virtual_penalty(config):
    reward = compute_reward(metrics(), metrics(), action_blocked=True, config=config)
    assert reward.blocked_action == config["reward"]["blocked_action_penalty"]
