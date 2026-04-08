"""Core environment implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from ..case_bank import get_case
from ..graders import score_final
from ..models import ActionType, CaseSpec, ProvisionalDecision, ResetRequest, StepResultModel, TriageAction, TriageObservation, TriageState
from ..rewards import TIMEOUT_PENALTY, compute_step_reward


class PatientTriageEnvironment:
    """Deterministic environment with a single in-memory episode."""

    def __init__(self) -> None:
        self._case: CaseSpec | None = None
        self._state: TriageState | None = None
        self._known_answers: dict[str, str] = {}
        self._provisional = ProvisionalDecision()
        self._done = False

    def reset(self, request: ResetRequest) -> StepResultModel:
        case = get_case(request.task, seed=request.seed, case_id=request.case_id)
        self._case = case
        self._known_answers = {}
        self._provisional = ProvisionalDecision()
        self._done = False
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=case.task_name,
            difficulty=case.difficulty,
            case_id=case.case_id,
            asked_questions=[],
            provisional_urgency=None,
            provisional_care_destination=None,
            captured_reason_codes=[],
            best_partial_score=0.0,
            final_score=None,
            action_history=[],
            last_feedback="Environment reset.",
            remaining_steps=case.max_steps,
        )
        observation = self._build_observation(
            reward=0.0,
            done=False,
            feedback="Environment reset and ready.",
            metadata={"reason_code_options": case.reason_code_options},
        )
        return StepResultModel(observation=observation, reward=0.0, done=False, info={"case_title": case.title})

    def step(self, action: TriageAction) -> StepResultModel:
        if self._case is None or self._state is None:
            raise RuntimeError("Environment must be reset before step()")
        if self._done:
            observation = self._build_observation(
                reward=0.0,
                done=True,
                feedback="Episode already finished. Call reset() to start a new case.",
                metadata={},
            )
            return StepResultModel(observation=observation, reward=0.0, done=True, info={"final_score": self._state.final_score})

        self._state.step_count += 1
        feedback = ""
        info: dict[str, Any] = {"reason_code_options": self._case.reason_code_options}
        question_outcome: str | None = None

        if action.action_type == ActionType.ASK_QUESTION:
            feedback, question_outcome = self._handle_question(action)
        elif action.action_type == ActionType.ASSIGN_URGENCY:
            self._provisional.urgency = action.urgency
            self._merge_reasoning(action)
            feedback = f"Recorded urgency assignment: {action.urgency.value}."
        elif action.action_type == ActionType.RECOMMEND_CARE:
            self._provisional.care_destination = action.care_destination
            self._merge_reasoning(action)
            feedback = f"Recorded care recommendation: {action.care_destination.value}."
        elif action.action_type == ActionType.FINALIZE:
            if action.urgency:
                self._provisional.urgency = action.urgency
            if action.care_destination:
                self._provisional.care_destination = action.care_destination
            self._merge_reasoning(action)
            feedback = "Final decision submitted."

        reward, new_best_score, reward_info = compute_step_reward(
            self._case,
            action.action_type,
            self._state.best_partial_score,
            self._decision_payload(),
            self._state.asked_questions,
            question_outcome=question_outcome,
        )
        self._state.best_partial_score = new_best_score
        self._sync_state_from_provisional()
        self._state.last_feedback = feedback
        self._state.remaining_steps = max(0, self._case.max_steps - self._state.step_count)
        self._state.action_history.append(
            {"step": self._state.step_count, "action": action.model_dump(mode="json"), "reward": reward}
        )

        done = False
        info.update({"partial_score": reward_info["partial_score"], "audit": reward_info["audit"]})

        if action.action_type == ActionType.FINALIZE:
            done = True
            final_score, final_audit = score_final(self._case, self._decision_payload(), self._state.asked_questions)
            self._state.final_score = final_score
            info["final_score"] = final_score
            info["audit"] = final_audit
            self._done = True
            reward = round(reward + (final_score - self._state.best_partial_score), 4)
            self._state.best_partial_score = max(self._state.best_partial_score, final_score)
            feedback = f"Episode finalized with score {final_score:.2f}."
            self._state.remaining_steps = 0
            self._state.last_feedback = feedback

        if not done and self._state.step_count >= self._case.max_steps:
            done = True
            final_score, final_audit = score_final(self._case, self._decision_payload(), self._state.asked_questions)
            self._state.final_score = final_score
            self._done = True
            reward = round(reward - TIMEOUT_PENALTY, 4)
            info["final_score"] = final_score
            info["audit"] = final_audit
            feedback = f"Step budget exhausted. Episode auto-finalized with score {final_score:.2f}."
            self._state.best_partial_score = max(self._state.best_partial_score, final_score)
            self._state.remaining_steps = 0
            self._state.last_feedback = feedback

        observation = self._build_observation(reward=reward, done=done, feedback=feedback, metadata=info)
        return StepResultModel(observation=observation, reward=reward, done=done, info=info)

    def state(self) -> TriageState:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet")
        return deepcopy(self._state)

    def _handle_question(self, action: TriageAction) -> tuple[str, str]:
        assert self._case is not None
        assert self._state is not None
        question_id = action.question_id or ""
        if question_id in self._state.asked_questions:
            return "Repeated clarifying question ignored.", "repeat"
        question = self._case.questions.get(question_id)
        if question is None:
            return "Invalid clarifying question.", "invalid"
        self._state.asked_questions.append(question_id)
        self._known_answers[question_id] = question.answer
        outcome = "useful" if question.useful else "irrelevant"
        return f"Question answered: {question.answer}", outcome

    def _merge_reasoning(self, action: TriageAction) -> None:
        if action.reason_codes:
            existing = set(self._provisional.reason_codes)
            for code in action.reason_codes:
                if code not in existing:
                    self._provisional.reason_codes.append(code)
        if action.rationale:
            self._provisional.rationale = action.rationale

    def _decision_payload(self) -> dict[str, Any]:
        return {
            "urgency": self._provisional.urgency,
            "care_destination": self._provisional.care_destination,
            "reason_codes": list(self._provisional.reason_codes),
            "rationale": self._provisional.rationale,
        }

    def _sync_state_from_provisional(self) -> None:
        assert self._state is not None
        self._state.provisional_urgency = self._provisional.urgency
        self._state.provisional_care_destination = self._provisional.care_destination
        self._state.captured_reason_codes = list(self._provisional.reason_codes)

    def _build_observation(self, *, reward: float, done: bool, feedback: str, metadata: dict[str, Any]) -> TriageObservation:
        assert self._case is not None
        assert self._state is not None
        return TriageObservation(
            done=done,
            reward=reward,
            task_name=self._case.task_name,
            difficulty=self._case.difficulty,
            case_id=self._case.case_id,
            patient_summary=self._case.patient_summary,
            known_answers=deepcopy(self._known_answers),
            allowed_question_ids=list(self._case.questions.keys()),
            remaining_steps=max(0, self._case.max_steps - self._state.step_count),
            available_actions=self._available_actions(done),
            provisional_decision=deepcopy(self._provisional),
            feedback=feedback,
            metadata=metadata,
        )

    def _available_actions(self, done: bool) -> list[ActionType]:
        if done:
            return []
        return [
            ActionType.ASK_QUESTION,
            ActionType.ASSIGN_URGENCY,
            ActionType.RECOMMEND_CARE,
            ActionType.FINALIZE,
        ]
