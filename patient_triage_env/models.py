"""Typed models for the patient triage environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State as OpenEnvState
except Exception:  # pragma: no cover
    class OpenEnvAction(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class OpenEnvObservation(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class OpenEnvState(BaseModel):
        model_config = ConfigDict(extra="forbid")
        episode_id: str = ""
        step_count: int = 0


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskName(str, Enum):
    URGENCY_CLASSIFICATION = "urgency_classification"
    CARE_RECOMMENDATION = "care_recommendation"
    FULL_TRIAGE_DECISION = "full_triage_decision"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CareDestination(str, Enum):
    HOME_CARE = "home_care"
    CLINIC = "clinic"
    URGENT_CARE = "urgent_care"
    EMERGENCY_ROOM = "emergency_room"


class ActionType(str, Enum):
    ASK_QUESTION = "ask_question"
    ASSIGN_URGENCY = "assign_urgency"
    RECOMMEND_CARE = "recommend_care"
    FINALIZE = "finalize"


class ProvisionalDecision(BaseModel):
    urgency: UrgencyLevel | None = None
    care_destination: CareDestination | None = None
    reason_codes: list[str] = Field(default_factory=list)
    rationale: str | None = None


class TriageAction(OpenEnvAction):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    question_id: str | None = None
    urgency: UrgencyLevel | None = None
    care_destination: CareDestination | None = None
    reason_codes: list[str] = Field(default_factory=list)
    rationale: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "TriageAction":
        if self.action_type == ActionType.ASK_QUESTION:
            if not self.question_id:
                raise ValueError("ask_question actions require question_id")
            if self.urgency or self.care_destination or self.reason_codes or self.rationale:
                raise ValueError("ask_question only supports question_id")
        if self.action_type == ActionType.ASSIGN_URGENCY:
            if not self.urgency:
                raise ValueError("assign_urgency requires urgency")
            if self.question_id or self.care_destination:
                raise ValueError("assign_urgency only supports urgency, optional reason_codes, optional rationale")
        if self.action_type == ActionType.RECOMMEND_CARE:
            if not self.care_destination:
                raise ValueError("recommend_care requires care_destination")
            if self.question_id or self.urgency:
                raise ValueError("recommend_care only supports care_destination, optional reason_codes, optional rationale")
        if self.action_type == ActionType.FINALIZE:
            if self.question_id:
                raise ValueError("finalize does not support question_id")
            if not any([self.urgency, self.care_destination, self.reason_codes, self.rationale]):
                raise ValueError("finalize requires at least one decision field")
        return self


class TriageObservation(OpenEnvObservation):
    model_config = ConfigDict(extra="forbid")

    done: bool = False
    reward: float = 0.0
    task_name: TaskName
    difficulty: Difficulty
    case_id: str
    patient_summary: str
    known_answers: dict[str, str] = Field(default_factory=dict)
    allowed_question_ids: list[str] = Field(default_factory=list)
    remaining_steps: int
    available_actions: list[ActionType] = Field(default_factory=list)
    provisional_decision: ProvisionalDecision = Field(default_factory=ProvisionalDecision)
    feedback: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TriageState(OpenEnvState):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    step_count: int = 0
    task_name: TaskName
    difficulty: Difficulty
    case_id: str
    asked_questions: list[str] = Field(default_factory=list)
    provisional_urgency: UrgencyLevel | None = None
    provisional_care_destination: CareDestination | None = None
    captured_reason_codes: list[str] = Field(default_factory=list)
    best_partial_score: float = 0.0
    final_score: float | None = None
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    last_feedback: str = ""
    remaining_steps: int = 0


class ClarifyingQuestion(BaseModel):
    prompt: str
    answer: str
    useful: bool = True


class CaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task_name: TaskName
    difficulty: Difficulty
    title: str
    patient_summary: str
    age_group: Literal["child", "adult", "older_adult", "pregnant"]
    symptoms: list[str]
    symptom_duration: str
    vitals: dict[str, str]
    history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    reason_code_options: list[str]
    questions: dict[str, ClarifyingQuestion] = Field(default_factory=dict)
    gold_urgency: UrgencyLevel
    gold_care_destination: CareDestination
    minimum_safe_urgency: UrgencyLevel
    minimum_safe_care_destination: CareDestination
    gold_reason_codes: list[str]
    mandatory_reason_codes: list[str] = Field(default_factory=list)
    rationale_keywords: list[str] = Field(default_factory=list)
    max_steps: int


class ResetRequest(BaseModel):
    task: TaskName = TaskName.URGENCY_CLASSIFICATION
    seed: int = 0
    case_id: str | None = None


class StepRequest(BaseModel):
    action: TriageAction


class StepResultModel(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StateEnvelope(BaseModel):
    state: TriageState


@dataclass(slots=True)
class StepResult:
    observation: TriageObservation
    reward: float
    done: bool
    info: dict[str, Any]
