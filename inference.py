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
TASK_SEQUENCE = [
    TaskName.URGENCY_CLASSIFICATION,
    TaskName.CARE_RECOMMENDATION,
    TaskName.FULL_TRIAGE_DECISION,
]

SYSTEM_PROMPT = """You are a careful benchmark agent for a synthetic patient triage environment.
Return only JSON with keys:
- action_type: ask_question | assign_urgency | recommend_care | finalize
- question_id: optional string
- urgency: optional low|medium|high|critical
- care_destination: optional home_care|clinic|urgent_care|emergency_room
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


def _strict_open_interval(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={model_name}")


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    error_value = error if error else "null"
    done_value = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}")


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    score = _strict_open_interval(sum(rewards) / len(rewards)) if rewards else 0.01
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")


def run() -> int:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for inference.py")

    env_url = os.environ.get("ENV_BASE_URL", DEFAULT_ENV_BASE_URL)
    api_base_url = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    requested_task = os.environ.get("TASK_NAME")
    base_seed = int(os.environ.get("TASK_SEED", "0"))

    env_client = PatientTriageEnv(base_url=env_url)
    llm_client = OpenAI(base_url=api_base_url, api_key=hf_token)
    exit_code = 1

    try:
        tasks_to_run = [TaskName(requested_task)] if requested_task else TASK_SEQUENCE
        all_success = True

        for index, task in enumerate(tasks_to_run):
            rewards: list[float] = []
            step_count = 0
            success = False
            started = False

            try:
                log_start(task.value, model_name)
                started = True
                try:
                    result = env_client.reset(task=task, seed=base_seed + index)
                except Exception as exc:
                    step_count = 1
                    rewards.append(0.01)
                    log_step(step_count, "reset", 0.01, True, _single_line(exc))
                    all_success = False
                    continue

                while True:
                    try:
                        action = choose_action(llm_client, model_name, result.observation)
                        result = env_client.step(action)
                        step_count += 1
                        rewards.append(float(result.reward))
                        action_str = json.dumps(action.model_dump(mode="json"), separators=(",", ":"))
                        error_str = _extract_last_action_error(result)
                        log_step(step_count, action_str, float(result.reward), bool(result.done), error_str)
                    except Exception as exc:
                        step_count += 1
                        rewards.append(0.0)
                        log_step(step_count, "error", 0.0, True, _single_line(exc))
                        all_success = False
                        break

                    if result.done:
                        final_score = _strict_open_interval(float(result.info.get("final_score", 0.5)))
                        success = final_score >= 0.7
                        if rewards:
                            rewards[-1] = final_score
                        else:
                            rewards.append(final_score)
                        if not success:
                            all_success = False
                        break
            finally:
                if started:
                    log_end(success=success, steps=step_count, rewards=rewards)

        exit_code = 0 if all_success else 1
    finally:
        env_client.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
