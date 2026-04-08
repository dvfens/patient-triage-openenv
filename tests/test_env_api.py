from fastapi.testclient import TestClient

from patient_triage_env.server.app import app


client = TestClient(app)


def test_root_endpoint_is_available():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert "/reset" in payload["endpoints"]


def test_reset_step_state_flow():
    reset_response = client.post("/reset", json={"task": "urgency_classification", "seed": 0})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["observation"]["task_name"] == "urgency_classification"
    assert reset_payload["done"] is False

    step_response = client.post(
        "/step",
        json={"action": {"action_type": "finalize", "urgency": "low"}},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["done"] is True
    assert 0.0 <= step_payload["info"]["final_score"] <= 1.0

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["state"]["final_score"] is not None


def test_seeded_reset_is_deterministic():
    first = client.post("/reset", json={"task": "care_recommendation", "seed": 3}).json()
    second = client.post("/reset", json={"task": "care_recommendation", "seed": 3}).json()
    assert first["observation"]["case_id"] == second["observation"]["case_id"]
