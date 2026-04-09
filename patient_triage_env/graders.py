"""Deterministic graders for all patient triage tasks."""

from __future__ import annotations

from typing import Any

from .models import CareDestination, CaseSpec, TaskName, UrgencyLevel


URGENCY_ORDER = {
    UrgencyLevel.LOW: 0,
    UrgencyLevel.MEDIUM: 1,
    UrgencyLevel.HIGH: 2,
    UrgencyLevel.CRITICAL: 3,
}

CARE_ORDER = {
    CareDestination.HOME_CARE: 0,
    CareDestination.CLINIC: 1,
    CareDestination.URGENT_CARE: 2,
    CareDestination.EMERGENCY_ROOM: 3,
}


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


def _urgency_score(case: CaseSpec, predicted: UrgencyLevel | None) -> float:
    if predicted is None:
        return 0.0
    delta = URGENCY_ORDER[predicted] - URGENCY_ORDER[case.gold_urgency]
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.8
    if delta == -1:
        if case.gold_urgency == UrgencyLevel.CRITICAL:
            return 0.0
        return 0.4
    return 0.0


def _care_score(case: CaseSpec, predicted: CareDestination | None) -> float:
    if predicted is None:
        return 0.0
    delta = CARE_ORDER[predicted] - CARE_ORDER[case.gold_care_destination]
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.75
    if delta == -1:
        if CARE_ORDER[case.minimum_safe_care_destination] >= CARE_ORDER[CareDestination.URGENT_CARE]:
            return 0.0
        return 0.2
    return 0.0


def _reason_code_score(case: CaseSpec, reason_codes: list[str]) -> float:
    if not case.gold_reason_codes:
        return 1.0
    provided = set(reason_codes)
    gold = set(case.gold_reason_codes)
    hits = len(provided & gold)
    return hits / len(gold)


def _mandatory_reason_penalty(case: CaseSpec, reason_codes: list[str]) -> bool:
    mandatory = set(case.mandatory_reason_codes)
    return bool(mandatory and not mandatory.issubset(set(reason_codes)))


def _rationale_score(case: CaseSpec, rationale: str | None) -> float:
    if not case.rationale_keywords:
        return 1.0
    if not rationale:
        return 0.0
    text = rationale.lower()
    hits = sum(1 for keyword in case.rationale_keywords if keyword.lower() in text)
    return hits / len(case.rationale_keywords)


def _question_score(asked_questions: list[str], useful_questions: set[str]) -> float:
    if not asked_questions:
        return 0.5
    if any(question_id in useful_questions for question_id in asked_questions):
        return 1.0
    return 0.0


def _is_dangerous_undertriage(
    case: CaseSpec,
    urgency: UrgencyLevel | None,
    care_destination: CareDestination | None,
    reason_codes: list[str],
) -> bool:
    if urgency is not None and URGENCY_ORDER[urgency] < URGENCY_ORDER[case.minimum_safe_urgency]:
        return True
    if care_destination is not None and CARE_ORDER[care_destination] < CARE_ORDER[case.minimum_safe_care_destination]:
        return True
    if _mandatory_reason_penalty(case, reason_codes):
        return True
    return False


def score_easy(case: CaseSpec, decision: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    urgency_score = _urgency_score(case, decision.get("urgency"))
    audit = {"task": TaskName.URGENCY_CLASSIFICATION.value, "urgency_score": urgency_score}
    return _clamp(urgency_score), audit


def score_medium(case: CaseSpec, decision: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    care_score = _care_score(case, decision.get("care_destination"))
    reason_score = _reason_code_score(case, decision.get("reason_codes", []))
    score = 0.8 * care_score + 0.2 * reason_score
    dangerous = _is_dangerous_undertriage(case, case.gold_urgency, decision.get("care_destination"), decision.get("reason_codes", []))
    if dangerous and care_score < 1.0:
        score = min(score, 0.2)
    audit = {
        "task": TaskName.CARE_RECOMMENDATION.value,
        "care_score": care_score,
        "reason_code_score": reason_score,
        "dangerous_undertriage": dangerous,
    }
    return _clamp(score), audit


def score_hard(case: CaseSpec, decision: dict[str, Any], asked_questions: list[str]) -> tuple[float, dict[str, Any]]:
    urgency_score = _urgency_score(case, decision.get("urgency"))
    care_score = _care_score(case, decision.get("care_destination"))
    reason_score = _reason_code_score(case, decision.get("reason_codes", []))
    useful_questions = {question_id for question_id, question in case.questions.items() if question.useful}
    question_score = _question_score(asked_questions, useful_questions)
    rationale_score = _rationale_score(case, decision.get("rationale"))
    dangerous = _is_dangerous_undertriage(
        case,
        decision.get("urgency"),
        decision.get("care_destination"),
        decision.get("reason_codes", []),
    )
    score = (
        0.30 * urgency_score
        + 0.35 * care_score
        + 0.15 * reason_score
        + 0.10 * question_score
        + 0.10 * rationale_score
    )
    if dangerous:
        score = min(score, 0.15)
    audit = {
        "task": TaskName.FULL_TRIAGE_DECISION.value,
        "urgency_score": urgency_score,
        "care_score": care_score,
        "reason_code_score": reason_score,
        "question_score": question_score,
        "rationale_score": rationale_score,
        "dangerous_undertriage": dangerous,
    }
    return _clamp(score), audit


def score_partial(case: CaseSpec, decision: dict[str, Any], asked_questions: list[str]) -> tuple[float, dict[str, Any]]:
    if case.task_name == TaskName.URGENCY_CLASSIFICATION:
        return score_easy(case, decision)
    if case.task_name == TaskName.CARE_RECOMMENDATION:
        return score_medium(case, decision)
    return score_hard(case, decision, asked_questions)


def score_final(case: CaseSpec, decision: dict[str, Any], asked_questions: list[str]) -> tuple[float, dict[str, Any]]:
    return score_partial(case, decision, asked_questions)
