"""Online learning on the real cluster takes hours, so it must survive a restart of the API."""
from core.state import ClusterMetrics
from engine.learning_engine import LearningEngine

METRICS = ClusterMetrics(cpu_percent=85.0, ram_percent=70.0, latency_ms=150.0, qps=800.0,
                         error_rate_percent=0.5, replicas=6)


def control_step(engine, mode):
    decision = engine.decide(mode, METRICS)
    engine.report(decision["decision_id"], "Hold", "baseline", False, "", 6, 6)


def test_online_learning_resumes_after_a_restart(config, tmp_path):
    data_dir, models_dir = str(tmp_path / "data"), str(tmp_path / "models")
    engine = LearningEngine(config, models_dir, data_dir)
    for _ in range(5):
        control_step(engine, "shadow")
    engine.save()
    assert engine.online_steps == 4    # the first call has no previous transition to learn from

    restarted = LearningEngine(config, models_dir, data_dir)
    assert restarted.online_steps == 4
    assert len(restarted.agent.replay) == 4
    # The learning curve is read back from data/online_learning.csv.
    curve = restarted.online_learning_curve()
    assert [point["online_steps"] for point in curve] == [1, 2, 3, 4]
    assert all(point["mode"] == "shadow" for point in curve)
