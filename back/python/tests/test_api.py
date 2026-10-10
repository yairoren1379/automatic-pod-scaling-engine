"""End-to-end test of the API: the calls the Go controller and the dashboard make."""
import importlib

import pytest
from fastapi.testclient import TestClient

METRICS = {"cpu_percent": 85.0, "ram_percent": 70.0, "latency_ms": 150.0, "qps": 800.0,
           "error_rate_percent": 0.5, "replicas": 6, "pod_crashed": False}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ZK_DISABLED", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))   # empty: untrained agent
    import api.server
    server = importlib.reload(api.server)
    with TestClient(server.app) as test_client:
        yield test_client


def control_step(client, mode, executed="Hold", blocked=False, metrics=METRICS):
    decision = client.post("/api/decide", json={"mode": mode, "metrics": metrics})
    assert decision.status_code == 200
    body = decision.json()
    assert body["action"] in ("ScaleUp", "ScaleDown", "Hold")
    report = client.post("/api/report", json={
        "decision_id": body["decision_id"], "executed_action": executed, "source": "baseline",
        "blocked": blocked, "block_reason": "max_replicas" if blocked else "",
        "replicas_before": metrics["replicas"], "replicas_after": metrics["replicas"]})
    assert report.status_code == 200
    return body


def test_decide_report_cycle_is_recorded(client):
    control_step(client, "baseline")
    control_step(client, "baseline", executed="ScaleUp", blocked=True)
    status = client.get("/api/status").json()
    assert status["mode"] == "baseline" and status["connected"]
    assert len(client.get("/api/history").json()) == 2
    audit = client.get("/api/audit").json()
    assert audit[-1]["blocked"] and audit[-1]["block_reason"] == "max_replicas"


def test_shadow_mode_learns_but_baseline_does_not(client):
    for _ in range(3):
        control_step(client, "baseline")
    assert client.get("/api/status").json()["online_steps"] == 0
    for _ in range(3):
        control_step(client, "shadow")
    # The first shadow call completes a transition that was executed in baseline mode.
    assert client.get("/api/status").json()["online_steps"] == 2


def test_report_for_an_old_decision_is_rejected(client):
    first = control_step(client, "shadow")
    control_step(client, "shadow")
    stale = client.post("/api/report", json={
        "decision_id": first["decision_id"], "executed_action": "Hold", "source": "dqn",
        "replicas_before": 6, "replicas_after": 6})
    assert stale.status_code == 409


def test_invalid_requests_are_rejected(client):
    assert client.post("/api/decide", json={"mode": "chaos", "metrics": METRICS}).status_code == 400
    assert client.post("/api/mode", json={"mode": "chaos"}).status_code == 400
    assert client.put("/api/config", json={"guardrails": {"min_replicas": 20}}).status_code == 400
    assert client.put("/api/config", json={"nope": {"x": 1}}).status_code == 400


def test_mode_config_and_load_go_through_the_store(client):
    assert client.post("/api/mode", json={"mode": "active"}).json() == {"mode": "active"}
    assert client.get("/api/config").json()["controller"]["mode"] == "active"

    updated = client.put("/api/config", json={"online": {"active_epsilon": 0.05}}).json()
    assert updated["online"]["active_epsilon"] == 0.05

    load = client.post("/api/load", json={"profile": "spike", "base_qps": 300}).json()
    assert load["profile"] == "spike" and load["base_qps"] == 300
    assert client.get("/api/load").json()["profile"] == "spike"


def test_websocket_sends_snapshot_then_steps(client):
    with client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot" and "config" in snapshot
        control_step(client, "shadow")
        event = ws.receive_json()
        while event["type"] != "step":
            event = ws.receive_json()
        assert event["point"]["qps"] == METRICS["qps"]
