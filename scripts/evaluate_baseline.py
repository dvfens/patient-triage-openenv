"""Run a baseline evaluation across all bundled cases."""

from __future__ import annotations

import json
import os
from statistics import mean

from patient_triage_env.case_bank import load_cases
from patient_triage_env.client import PatientTriageEnv
from patient_triage_env.models import ActionType, TaskName, TriageAction


def heuristic_action(observation) -> TriageAction:
    summary = observation.patient_summary.lower()
    known_text = " ".join(observation.known_answers.values()).lower()
    full_text = f"{summary} {known_text}"

    reason_codes = []
    urgency = None
    care = None

    if any(
        token in full_text
        for token in ["anaphylaxis", "blue lips", "suicidal", "arm weakness", "facial droop", "oxygen 88", "severe chest pressure", "black tarry", "shock"]
    ):
        urgency = "critical"
        care = "emergency_room"
        reason_codes = ["red_flag_symptom", "abnormal_vitals"]
    elif any(token in full_text for token in ["chest pain", "stroke", "blood thinner", "pregnant", "rebound tenderness", "wheeze", "low oxygen"]):
        urgency = "high"
        care = "urgent_care" if "wheeze" in full_text and "oxygen 88" not in full_text else "emergency_room"
        reason_codes = ["red_flag_symptom"]
    elif any(token in full_text for token in ["fever", "dehydration", "elderly", "tachycardia", "asthma flare", "persistent vomiting"]):
        urgency = "medium"
        care = "clinic_visit"
        reason_codes = ["persistent_or_worsening_symptoms"]
    else:
        urgency = "low"
        care = "home_care"
        reason_codes = ["stable_for_outpatient_follow_up"]

    if observation.task_name == TaskName.URGENCY_CLASSIFICATION:
        return TriageAction(action_type=ActionType.FINALIZE, urgency=urgency)
    if observation.task_name == TaskName.CARE_RECOMMENDATION:
        return TriageAction(
            action_type=ActionType.FINALIZE,
            care_destination=care,
            reason_codes=reason_codes,
            rationale="Rule-based baseline based on symptom severity and risk modifiers.",
        )
    return TriageAction(
        action_type=ActionType.FINALIZE,
        urgency=urgency,
        care_destination=care,
        reason_codes=reason_codes,
        rationale="Rule-based baseline based on symptom severity, red flags, and vital sign risk.",
    )


def evaluate(base_url: str = "http://localhost:8000") -> dict[str, object]:
    client = PatientTriageEnv(base_url=base_url)
    scores_by_task: dict[str, list[float]] = {}
    for task in TaskName:
        scores: list[float] = []
        for case in load_cases(task):
            result = client.reset(task=task, case_id=case.case_id, seed=0)
            action = heuristic_action(result.observation)
            final_result = client.step(action)
            scores.append(float(final_result.info.get("final_score", 0.0)))
        scores_by_task[task.value] = scores
    client.close()
    return {
        "per_task_average": {task: round(mean(scores), 4) for task, scores in scores_by_task.items()},
        "overall_average": round(mean(score for scores in scores_by_task.values() for score in scores), 4),
    }


if __name__ == "__main__":
    report = evaluate(base_url=os.environ.get("ENV_BASE_URL", "http://localhost:8000"))
    print(json.dumps(report, indent=2))
