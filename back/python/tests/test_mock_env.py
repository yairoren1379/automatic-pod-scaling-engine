from core.actions import HOLD, SCALE_DOWN, SCALE_UP
from simulation.load_profiles import LoadProfile
from simulation.mock_env import MockKubernetesEnv


def make_env(config, qps=400, replicas=6):
    env = MockKubernetesEnv(config, seed=1, profile_kind="constant")
    env.reset()
    env.set_profile(LoadProfile("constant", qps))
    env.qps, env.qps_trend = qps, 0.0
    env.ready_pods, env.starting_pods, env.warming_pods, env.recovering_pods = replicas, [], [], []
    config["mock_env"]["random_spike_probability"] = 0.0
    config["mock_env"]["random_crash_probability"] = 0.0
    env.ram = env._ram_target(env._utilization())
    env.metrics = env._measure(scale_down_shock=0.0, termination_errors=0.0, allow_random_events=False)
    return env


def test_same_seed_gives_same_episode(config):
    runs = []
    for _ in range(2):
        env = MockKubernetesEnv(config, seed=42)
        env.reset()
        runs.append([env.step(HOLD)[0] for _ in range(20)])
    assert runs[0] == runs[1]


def test_guardrails_block_beyond_limits(config):
    env = make_env(config, replicas=config["guardrails"]["max_replicas"])
    _, _, _, _, info = env.step(SCALE_UP)
    assert info["blocked"]
    assert info["reward"]["blocked_action"] == config["reward"]["blocked_action_penalty"]

    env = make_env(config, replicas=config["guardrails"]["min_replicas"])
    assert env.step(SCALE_DOWN)[4]["blocked"]


def test_scale_down_is_blocked_right_after_a_scale_up(config):
    env = make_env(config, qps=300, replicas=6)
    window = env.stabilization_steps
    assert window == 5   # 150 s / 30 s per loop
    env.step(SCALE_UP)
    for _ in range(window - 1):
        _, _, _, _, info = env.step(SCALE_DOWN)
        assert info["blocked"] and info["block_reason"] == "stabilization_window"
    _, _, _, _, info = env.step(SCALE_DOWN)
    assert not info["blocked"]
    # Scaling up is never delayed.
    assert not env.step(SCALE_UP)[4]["blocked"]


def test_new_pod_starts_cold_then_takes_its_full_share(config):
    env = make_env(config, qps=540, replicas=6)
    cpu_before = env.metrics.cpu_percent
    first, *_ = env.step(SCALE_UP)
    assert first.replicas == 7
    later = [env.step(HOLD)[0] for _ in range(3)][-1]
    # Ready by the next loop but still cold (reduced capacity), then fully warm.
    assert later.cpu_percent < first.cpu_percent < cpu_before


def test_scale_down_drops_requests(config):
    env = make_env(config, qps=200, replicas=6)
    metrics, *_ = env.step(SCALE_DOWN)
    assert metrics.error_rate_percent > 0


def test_overload_causes_errors_and_latency(config):
    env = make_env(config, qps=1200, replicas=4)
    metrics = [env.step(HOLD)[0] for _ in range(3)][-1]
    assert metrics.error_rate_percent > 10
    assert metrics.latency_ms > config["reward"]["latency_slo_ms"]
