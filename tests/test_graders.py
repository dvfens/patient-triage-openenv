from patient_triage_env.case_bank import get_case
from patient_triage_env.graders import score_easy, score_hard, score_medium
from patient_triage_env.models import CareDestination, TaskName, UrgencyLevel


def test_easy_grader_rewards_exact_match():
    case = get_case(TaskName.URGENCY_CLASSIFICATION, case_id="easy_01_uri")
    score, audit = score_easy(case, {"urgency": UrgencyLevel.LOW})
    assert score == 1.0
    assert audit["urgency_score"] == 1.0


def test_medium_grader_penalizes_undertriage():
    case = get_case(TaskName.CARE_RECOMMENDATION, case_id="medium_05_abdominal_pain")
    safe_score, _ = score_medium(case, {"care_destination": CareDestination.EMERGENCY_ROOM, "reason_codes": ["red_flag_symptom"]})
    unsafe_score, audit = score_medium(case, {"care_destination": CareDestination.CLINIC, "reason_codes": []})
    assert safe_score > unsafe_score
    assert audit["dangerous_undertriage"] is True


def test_hard_grader_caps_dangerous_miss():
    case = get_case(TaskName.FULL_TRIAGE_DECISION, case_id="hard_01_chest_pain")
    score, audit = score_hard(
        case,
        {
            "urgency": UrgencyLevel.MEDIUM,
            "care_destination": CareDestination.CLINIC,
            "reason_codes": [],
            "rationale": "This sounds stable.",
        },
        asked_questions=[],
    )
    assert score <= 0.15
    assert audit["dangerous_undertriage"] is True
