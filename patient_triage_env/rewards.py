"""Reward shaping for patient triage episodes."""

from __future__ import annotations

from .graders import score_partial
from .models import ActionType, CaseSpec


QUESTION_BONUS = 0.10
IRRELEVANT_QUESTION_PENALTY = 0.05
INVALID_QUESTION_PENALTY = 0.10
REPEAT_QUESTION_PENALTY = 0.10
FINALIZE_WITHOUT_DECISION_PENALTY = 0.10
TIMEOUT_PENALTY = 0.10


def compute_step_reward(
    case: CaseSpec,
    action_type: ActionType,
    previous_best_score: float,
    current_decision: dict[str, object],
    asked_questions: list[str],
    *,
    question_outcome: str | None = None,
) -> tuple[float, float, dict[str, object]]:
    partial_score, audit = score_partial(case, current_decision, asked_questions)
    new_best_score = max(previous_best_score, partial_score)
    delta = new_best_score - previous_best_score

    penalty = 0.0
    if action_type == ActionType.ASK_QUESTION:
        if question_outcome == "useful":
            delta += QUESTION_BONUS
        elif question_outcome == "irrelevant":
            penalty += IRRELEVANT_QUESTION_PENALTY
        elif question_outcome == "invalid":
            penalty += INVALID_QUESTION_PENALTY
        elif question_outcome == "repeat":
            penalty += REPEAT_QUESTION_PENALTY

    if action_type == ActionType.FINALIZE and partial_score == 0.0:
        penalty += FINALIZE_WITHOUT_DECISION_PENALTY

    reward = round(delta - penalty, 4)
    return reward, new_best_score, {"partial_score": partial_score, "penalty": penalty, "audit": audit}
