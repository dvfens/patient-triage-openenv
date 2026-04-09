import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import inference
from patient_triage_env.models import ActionType, Difficulty, ProvisionalDecision, TaskName


class FakeEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.calls = 0
        self.closed = False

    def reset(self, task, seed):
        observation = SimpleNamespace(
            task_name=task,
            difficulty=Difficulty.HARD,
            case_id="hard_01_chest_pain",
            patient_summary="Synthetic summary",
            known_answers={},
            allowed_question_ids=[],
            remaining_steps=5,
            available_actions=[ActionType.FINALIZE],
            provisional_decision=ProvisionalDecision(),
            feedback="",
            metadata={"reason_code_options": ["red_flag_symptom"]},
            model_dump=lambda mode="json": {
                "task_name": task.value,
                "case_id": "hard_01_chest_pain",
                "patient_summary": "Synthetic summary",
            },
        )
        return SimpleNamespace(observation=observation, reward=0.0, done=False, info={})

    def step(self, action):
        self.calls += 1
        observation = SimpleNamespace(
            task_name=TaskName.FULL_TRIAGE_DECISION,
            difficulty=Difficulty.HARD,
            case_id="hard_01_chest_pain",
            patient_summary="Synthetic summary",
            known_answers={},
            allowed_question_ids=[],
            remaining_steps=0,
            available_actions=[],
            provisional_decision=ProvisionalDecision(urgency=action.urgency, care_destination=action.care_destination),
            feedback="done",
            metadata={"last_action_error": None},
            model_dump=lambda mode="json": {},
        )
        return SimpleNamespace(observation=observation, reward=1.0, done=True, info={"final_score": 1.0})

    def close(self):
        self.closed = True
        return None


class FakeCompletionClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                content = (
                    '{"action_type":"finalize","urgency":"critical",'
                    '"care_destination":"emergency_room","reason_codes":["red_flag_symptom"],'
                    '"rationale":"Emergency evaluation required."}'
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_inference_print_format(monkeypatch):
    fake_client = FakeEnvClient("http://localhost:8000")
    monkeypatch.setattr(inference, "PatientTriageEnv", lambda base_url: fake_client)
    monkeypatch.setattr(inference, "OpenAI", lambda **kwargs: FakeCompletionClient())
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("TASK_NAME", "full_triage_decision")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        inference.run()

    output = stdout.getvalue().strip().splitlines()
    assert output[0].startswith("[START] task=full_triage_decision")
    assert output[1].startswith("[STEP] step=1 action=")
    assert " reward=1.00 " in output[1]
    assert " done=true " in output[1]
    assert output[1].endswith("error=null")
    assert output[2].startswith("[END] success=true steps=1 rewards=1.00")
    assert fake_client.closed is True
