"""Typed synchronous client for the patient triage environment."""

from __future__ import annotations

from typing import Any

import requests

from .models import ResetRequest, StateEnvelope, StepRequest, StepResult, StepResultModel, TaskName, TriageAction, TriageState


class PatientTriageEnv:
    """Simple HTTP client that mirrors the OpenEnv reset/step/state flow."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/healthz", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def reset(self, task: TaskName = TaskName.URGENCY_CLASSIFICATION, seed: int = 0, case_id: str | None = None) -> StepResult:
        payload = ResetRequest(task=task, seed=seed, case_id=case_id)
        response = self.session.post(f"{self.base_url}/reset", json=payload.model_dump(mode="json"), timeout=self.timeout_s)
        response.raise_for_status()
        parsed = StepResultModel.model_validate(response.json())
        return StepResult(observation=parsed.observation, reward=parsed.reward, done=parsed.done, info=parsed.info)

    def step(self, action: TriageAction) -> StepResult:
        payload = StepRequest(action=action)
        response = self.session.post(f"{self.base_url}/step", json=payload.model_dump(mode="json"), timeout=self.timeout_s)
        response.raise_for_status()
        parsed = StepResultModel.model_validate(response.json())
        return StepResult(observation=parsed.observation, reward=parsed.reward, done=parsed.done, info=parsed.info)

    def state(self) -> TriageState:
        response = self.session.get(f"{self.base_url}/state", timeout=self.timeout_s)
        response.raise_for_status()
        parsed = StateEnvelope.model_validate(response.json())
        return parsed.state
