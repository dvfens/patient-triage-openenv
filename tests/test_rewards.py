from patient_triage_env.case_bank import get_case
from patient_triage_env.models import ActionType, TaskName, UrgencyLevel
from patient_triage_env.rewards import compute_step_reward


def test_useful_question_gets_positive_reward():
    case = get_case(TaskName.FULL_TRIAGE_DECISION, case_id="hard_02_stroke")
    reward, new_best, info = compute_step_reward(
        case,
        ActionType.ASK_QUESTION,
        0.0,
        {"urgency": None, "care_destination": None, "reason_codes": [], "rationale": None},
        asked_questions=["last_known_well"],
        question_outcome="useful",
    )
    assert reward > 0
    assert new_best >= 0
    assert "audit" in info


def test_correct_urgency_improves_partial_score():
    case = get_case(TaskName.URGENCY_CLASSIFICATION, case_id="easy_06_mild_asthma")
    reward, new_best, _ = compute_step_reward(
        case,
        ActionType.ASSIGN_URGENCY,
        0.0,
        {"urgency": UrgencyLevel.MEDIUM, "care_destination": None, "reason_codes": [], "rationale": None},
        asked_questions=[],
    )
    assert reward > 0
    assert new_best > 0
