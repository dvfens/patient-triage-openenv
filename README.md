---
title: Patient Triage OpenEnv
emoji: "🏥"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Patient Triage OpenEnv

Deterministic synthetic patient-triage benchmark built for OpenEnv-style agent evaluation. The environment is a real-world workflow benchmark, not a toy game: an agent must assess structured patient presentations, ask at most one clarifying question when allowed, assign urgency, recommend a care destination, and justify its decision with controlled reason codes.

## Requirement Checklist

- `reset()`, `step(action)`, and `state()` are implemented as typed APIs in [patient_triage_env/server/environment.py](/C:/Users/Lenovo/Documents/metascaler/patient_triage_env/server/environment.py).
- The environment models a practical task: first-line triage and care routing for synthetic patient cases.
- Three tasks are included with increasing difficulty:
  - `urgency_classification` (easy)
  - `care_recommendation` (medium)
  - `full_triage_decision` (hard)
- The grader returns a meaningful score from `0.0` to `1.0` with auditable components in [patient_triage_env/graders.py](/C:/Users/Lenovo/Documents/metascaler/patient_triage_env/graders.py).
- Reward is shaped step by step in [patient_triage_env/rewards.py](/C:/Users/Lenovo/Documents/metascaler/patient_triage_env/rewards.py), including penalties for unsafe and low-value actions.
- A baseline agent lives in [inference.py](/C:/Users/Lenovo/Documents/metascaler/inference.py) and interacts with the environment end-to-end.
- Docker packaging is provided via [Dockerfile](/C:/Users/Lenovo/Documents/metascaler/Dockerfile).
- Hugging Face Space/OpenEnv metadata is provided via [openenv.yaml](/C:/Users/Lenovo/Documents/metascaler/openenv.yaml).

## Why This Satisfies The Challenge

- Real-world utility: triage quality, routing safety, and auditability are all easy for judges to understand.
- Strong environment design: actions, observations, rewards, and grader logic are explicit and deterministic.
- Clear tasks: each difficulty increases the action space and evaluation complexity without changing the domain.
- Reliable evaluation: synthetic cases, pinned dependencies, and no environment-side external services keep runs reproducible.

## Environment Design

### Observation Space

Each observation includes:

- `patient_summary`
- `known_answers`
- `allowed_question_ids`
- `remaining_steps`
- `available_actions`
- `provisional_decision`
- `metadata.reason_code_options`

### Action Space

Agents submit a typed `TriageAction` with:

- `action_type`: `ask_question`, `assign_urgency`, `recommend_care`, or `finalize`
- optional `question_id`
- optional `urgency`
- optional `care_destination`
- optional `reason_codes`
- optional `rationale`

### Tasks

`urgency_classification`
- Goal: assign the correct urgency level.
- Cases are fully specified and usually do not benefit from questions.

`care_recommendation`
- Goal: route the patient to the right care setting.
- Safe over-triage gets partial credit; unsafe under-triage is heavily penalized.

`full_triage_decision`
- Goal: optionally ask one clarifying question, then decide urgency, care destination, reason codes, and concise rationale.
- Hard cases include chest pain, stroke symptoms, GI bleed, anaphylaxis, suicidal risk, sepsis, hypoxic COPD, and pregnancy red flags.

## Scoring And Reward

The grader is deterministic and auditable:

- Easy: urgency scoring with exact, safe-over, and under-triage rules.
- Medium: weighted care-destination plus controlled reason-code coverage.
- Hard: weighted urgency, destination, reason codes, question usefulness, and rationale completeness.
- Safety override: dangerous misses cap the score.

Reward is shaped:

- Positive reward for useful clarification and correct progress.
- Negative reward for repeated or invalid questions.
- Stronger penalties for unsafe under-triage.
- Timeout penalty when the step budget is exhausted.

## Case Bank

The synthetic dataset ships with 24 cases:

- 8 easy cases in [patient_triage_env/data/easy_cases.json](/C:/Users/Lenovo/Documents/metascaler/patient_triage_env/data/easy_cases.json)
- 8 medium cases in [patient_triage_env/data/medium_cases.json](/C:/Users/Lenovo/Documents/metascaler/patient_triage_env/data/medium_cases.json)
- 8 hard cases in [patient_triage_env/data/hard_cases.json](/C:/Users/Lenovo/Documents/metascaler/patient_triage_env/data/hard_cases.json)

All cases are synthetic. This benchmark is not medical advice and should be presented as an evaluation environment, not a clinical system.

## Local Setup

```bash
pip install -r requirements.txt
uvicorn patient_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

## Example Usage

```python
from patient_triage_env.client import PatientTriageEnv
from patient_triage_env.models import ActionType, TaskName, TriageAction

env = PatientTriageEnv("http://localhost:8000")
result = env.reset(task=TaskName.FULL_TRIAGE_DECISION, seed=0)

action = TriageAction(
    action_type=ActionType.FINALIZE,
    urgency="critical",
    care_destination="emergency_room",
    reason_codes=["red_flag_symptom", "abnormal_vitals"],
    rationale="Emergency evaluation required due to red flags and unstable vitals.",
)

result = env.step(action)
print(result.info["final_score"])
env.close()
```

## Baseline Agent

The submission baseline is [inference.py](/C:/Users/Lenovo/Documents/metascaler/inference.py). It:

- reads `API_BASE_URL`, `MODEL_NAME`, and required `HF_TOKEN`
- connects to the local environment using `ENV_BASE_URL`
- calls an OpenAI-compatible chat endpoint
- prints exact `[START]`, `[STEP]`, and `[END]` lines for evaluation logs

Run it with:

```bash
HF_TOKEN=your_token_here python inference.py
```

Defaults:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=deepseek-ai/DeepSeek-V3-0324`
- `ENV_BASE_URL=http://localhost:8000`
- `TASK_NAME=full_triage_decision`
- `TASK_SEED=0`

## Baseline Evaluation

Use [scripts/evaluate_baseline.py](/C:/Users/Lenovo/Documents/metascaler/scripts/evaluate_baseline.py) to compute submission numbers after the server is running:

```bash
python scripts/evaluate_baseline.py
```

The output reports:

- average score per task
- overall average across all bundled cases

## Docker

```bash
docker build -t patient-triage-openenv .
docker run --rm -p 8000:8000 patient-triage-openenv
```

## Hugging Face Space Notes

- The app entrypoint is `patient_triage_env.server.app:app`.
- The root `Dockerfile` is Space-ready.
- `openenv.yaml` is included for OpenEnv/HF deployment metadata.
- The benchmark does not require environment-side network access.

## Tests

```bash
pytest
```

Current test coverage includes:

- reset/step/state API flow
- deterministic case selection by seed
- grader behavior on safe vs unsafe decisions
- reward shaping for useful questions and correct progress
- `inference.py` stdout formatting
