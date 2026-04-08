"""Case loading and deterministic selection helpers."""

from __future__ import annotations

import json
from importlib import resources

from .models import CaseSpec, TaskName


TASK_FILE_MAP = {
    TaskName.URGENCY_CLASSIFICATION: "easy_cases.json",
    TaskName.CARE_RECOMMENDATION: "medium_cases.json",
    TaskName.FULL_TRIAGE_DECISION: "hard_cases.json",
}


def load_cases(task_name: TaskName) -> list[CaseSpec]:
    file_name = TASK_FILE_MAP[task_name]
    data_text = resources.files("patient_triage_env.data").joinpath(file_name).read_text(encoding="utf-8")
    raw_cases = json.loads(data_text)
    return [CaseSpec.model_validate(item) for item in raw_cases]


def get_case(task_name: TaskName, seed: int = 0, case_id: str | None = None) -> CaseSpec:
    cases = load_cases(task_name)
    if case_id:
        for case in cases:
            if case.case_id == case_id:
                return case
        raise ValueError(f"Unknown case_id '{case_id}' for task '{task_name.value}'")
    index = abs(seed) % len(cases)
    return sorted(cases, key=lambda case: case.case_id)[index]
