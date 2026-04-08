"""Baseline agent runner for the patient triage benchmark."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from patient_triage_env.client import PatientTriageEnv
from patient_triage_env.models import TaskName, TriageAction


BENCHMARK_NAME = "patient-triage-openenv"
DEFAULT_ENV_BASE_URL = "http://localhost:8000"
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"

SYSTEM_PROMPT = """You are a careful benchmark agent for a synthetic patient triage environment.
Return only JSON with keys:
- action_type: ask_question | assign_urgency | recommend_care | finalize
- question_id: optional string
- urgency: optional low|medium|high|critical
- care_destination: optional home_care|clinic_visit|urgent_care|emergency_room
- reason_codes: optional array of strings from the observation metadata
- rationale: optional concise string
Be safety-aware. Prefer concise, structured outputs."""


def build_user_prompt(observation: Any) -> str:
    metadata = observation.metadata or {}
    reason_codes = metadata.get("reason_code_options", [])
    return (
        "Current observation:\n"
        f"{json.dumps(observation.model_dump(mode='json'), indent=2)}\n\n"
        "Choose the next single action.\n"
        f"Available reason codes: {reason_codes}\n"
        "If enough information is available, finalize."
    )


def choose_action(client: OpenAI, model_name: str, observation: Any) -> TriageAction:
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(observation)},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    return TriageAction.model_validate(payload)


def _single_line(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _extract_last_action_error(result: Any) -> str:
    info = getattr(result, "info", {}) or {}
    observation = getattr(result, "observation", None)
    metadata = getattr(observation, "metadata", {}) or {}

    error = metadata.get("last_action_error")
    if error is None:
        error = info.get("last_action_error")
    if error is None:
        error = info.get("error")
    if error in (None, ""):
        return "null"
    return _single_line(error)


def run() -> int:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for inference.py")

    env_url = os.environ.get("ENV_BASE_URL", DEFAULT_ENV_BASE_URL)
    api_base_url = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    task = TaskName(os.environ.get("TASK_NAME", TaskName.FULL_TRIAGE_DECISION.value))
    seed = int(os.environ.get("TASK_SEED", "0"))

    env_client = PatientTriageEnv(base_url=env_url)
    llm_client = OpenAI(base_url=api_base_url, api_key=hf_token)
    rewards: list[str] = []
    step_count = 0
    success = False
    started = False
    exit_code = 1

    try:
        result = env_client.reset(task=task, seed=seed)
        print(f"[START] task={task.value} env={BENCHMARK_NAME} model={model_name}")
        started = True

        while True:
            action = choose_action(llm_client, model_name, result.observation)
            result = env_client.step(action)
            step_count += 1
            rewards.append(f"{result.reward:.2f}")
            action_str = json.dumps(action.model_dump(mode="json"), separators=(",", ":"))
            error_str = _extract_last_action_error(result)
            print(
                f"[STEP] step={step_count} action={action_str} reward={result.reward:.2f} "
                f"done={str(result.done).lower()} error={error_str}"
            )
            if result.done:
                success = bool(result.info.get("final_score", 0.0) >= 0.7)
                exit_code = 0
                break
    finally:
        env_client.close()
        if started:
            print(f"[END] success={str(success).lower()} steps={step_count} rewards={','.join(rewards)}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
