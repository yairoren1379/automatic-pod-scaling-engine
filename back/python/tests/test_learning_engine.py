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


def test_a_fresh_clone_starts_from_the_shipped_cluster_model(config, tmp_path):
    # models/online_dqn.pt is the model fine-tuned on the real cluster; with no data/ yet (a
    # fresh git clone), the engine starts from it, and keeps learning in data/.
    trained = LearningEngine(config, str(tmp_path / "models"), str(tmp_path / "trained_data"))
    for _ in range(3):
        control_step(trained, "shadow")
    (tmp_path / "models").mkdir()
    trained.agent.save(str(tmp_path / "models" / "online_dqn.pt"), extra={"online_steps": 2})
    trained.agent.replay.save(str(tmp_path / "models" / "online_replay.pkl"))

    fresh = LearningEngine(config, str(tmp_path / "models"), str(tmp_path / "data"))
    assert fresh.model_source.startswith("fine-tuned on the real cluster")
    assert fresh.online_steps == 2 and len(fresh.agent.replay) == 2
