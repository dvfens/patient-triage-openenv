"""FastAPI app exposing reset, step, state, and WebSocket APIs."""

from __future__ import annotations

import re

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..models import ResetRequest, StateEnvelope, StepRequest
from .environment import PatientTriageEnvironment


app = FastAPI(
    title="Patient Triage OpenEnv",
    version="0.1.0",
    description="Synthetic patient triage benchmark for agent evaluation.",
)
env = PatientTriageEnvironment()


def _normalize_message(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _chat_triage_suggestion(message: str) -> dict[str, object]:
    normalized = _normalize_message(message)
    if not normalized:
        raise ValueError("Please describe symptoms, a disease, or a triage concern.")

    urgency = "medium"
    care_destination = "clinic"
    reason_codes: list[str] = ["persistent_or_worsening_symptoms"]
    label = "Needs timely medical review"
    rationale = "Symptoms suggest an in-person evaluation is safer than self-management."

    emergency_signals = [
        "chest pain",
        "stroke",
        "face drooping",
        "slurred speech",
        "anaphylaxis",
        "throat swelling",
        "severe bleeding",
        "unconscious",
        "passed out",
        "blue lips",
        "suicidal",
        "seizure",
        "oxygen 88",
        "oxygen 89",
        "oxygen 90",
    ]
    urgent_signals = [
        "shortness of breath",
        "asthma attack",
        "dehydration",
        "high fever",
        "pregnant bleeding",
        "abdominal pain",
        "severe headache",
        "worsening cough",
        "panic with chest tightness",
    ]
    mild_signals = [
        "common cold",
        "runny nose",
        "sore throat",
        "mild cough",
        "seasonal allergy",
        "mild headache",
    ]

    if _contains_any(normalized, emergency_signals) or (
        "chest pain" in normalized and _contains_any(normalized, ["sweating", "left arm", "jaw", "breathless"])
    ):
        urgency = "critical"
        care_destination = "emergency_room"
        reason_codes = ["red_flag_symptom", "abnormal_vitals", "persistent_or_worsening_symptoms"]
        label = "Emergency care now"
        rationale = "This pattern includes red-flag features that should be assessed urgently in an emergency setting."
    elif _contains_any(normalized, urgent_signals) or (
        "fever" in normalized and _contains_any(normalized, ["infant", "baby", "older adult", "elderly"])
    ):
        urgency = "high"
        care_destination = "urgent_care"
        reason_codes = ["red_flag_symptom", "high_risk_history", "persistent_or_worsening_symptoms"]
        label = "Urgent same-day assessment"
        rationale = "The description suggests elevated risk or worsening symptoms that deserve prompt evaluation."
    elif _contains_any(normalized, mild_signals):
        urgency = "low"
        care_destination = "home_care"
        reason_codes = ["persistent_or_worsening_symptoms"]
        label = "Likely home care with monitoring"
        rationale = "The symptoms sound mild and may be suitable for supportive care if no red flags are present."
    elif _contains_any(normalized, ["uti", "migraine", "vomiting", "diarrhea", "ear pain", "rash"]):
        urgency = "medium"
        care_destination = "clinic"
        reason_codes = ["persistent_or_worsening_symptoms", "red_flag_symptom"]
        label = "Clinic visit recommended"
        rationale = "The presentation could require treatment or testing but does not sound immediately life-threatening."

    answer = (
        f"{label}. Suggested urgency: {urgency}. Suggested care destination: {care_destination}. "
        f"Key reason codes: {', '.join(reason_codes)}."
    )

    return {
        "message": message,
        "answer": answer,
        "urgency": urgency,
        "care_destination": care_destination,
        "reason_codes": reason_codes,
        "rationale": rationale,
        "suggested_action": {
            "action_type": "finalize",
            "urgency": urgency,
            "care_destination": care_destination,
            "reason_codes": reason_codes,
            "rationale": rationale,
        },
        "disclaimer": "This is a deterministic benchmark-style helper, not medical advice.",
    }


@app.get("/")
def root() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Patient Triage OpenEnv</title>
    <style>
        :root {
            --bg: #f7f1f5;
            --panel: rgba(255, 255, 255, 0.34);
            --panel-soft: rgba(255, 255, 255, 0.52);
            --border: rgba(88, 83, 114, 0.08);
            --ink: #2f2b38;
            --muted: #7f768a;
            --accent: #ff9c88;
            --accent-2: #f3a9d8;
            --warn: #ff8b6f;
            --danger: #ff6b7f;
            --shadow: 0 18px 45px rgba(120, 102, 132, 0.08);
            --radius-xl: 32px;
            --radius-lg: 24px;
            --radius-md: 18px;
            --radius-sm: 16px;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Aptos, "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(900px 620px at 0% 10%, rgba(194, 211, 255, 0.6) 0%, transparent 55%),
                radial-gradient(980px 680px at 100% 0%, rgba(255, 213, 220, 0.56) 0%, transparent 52%),
                radial-gradient(680px 420px at 50% 100%, rgba(255, 236, 207, 0.45) 0%, transparent 60%),
                var(--bg);
            min-height: 100vh;
            padding: 26px 18px 44px;
        }
        .wrap {
            max-width: 1080px;
            margin: 0 auto;
            display: grid;
            gap: 18px;
        }
        .hero,
        .panel,
        .response-shell {
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: var(--radius-xl);
            background:
                linear-gradient(160deg, rgba(255, 255, 255, 0.62), rgba(255, 255, 255, 0.2)),
                var(--panel);
            backdrop-filter: blur(18px);
            box-shadow: var(--shadow);
        }
        .hero {
            padding: 42px 24px 18px;
            display: grid;
            justify-items: center;
            text-align: center;
            gap: 14px;
            background: transparent;
            border: none;
            box-shadow: none;
            backdrop-filter: none;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.5);
            color: #6f647a;
            border: 1px solid rgba(126, 117, 140, 0.08);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.5rem, 6vw, 5rem);
            line-height: 1.05;
            letter-spacing: -0.05em;
            margin-bottom: 6px;
        }
        .hero p {
            margin: 0;
            max-width: 560px;
            margin-inline: auto;
            text-align: center;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
        }
        .hero-meta {
            display: none;
        }
        .hero-stat {
            padding: 16px;
            border-radius: var(--radius-md);
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(10, 21, 38, 0.58);
        }
        .hero-stat strong {
            display: block;
            font-size: 1.35rem;
            margin-bottom: 4px;
        }
        .hero-stat span {
            color: var(--muted);
            font-size: 0.88rem;
        }
        .workspace {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            align-items: start;
        }
        @media (max-width: 1180px) {
            .workspace {
                grid-template-columns: 1fr;
            }
        }
        .panel {
            padding: 16px 18px;
            background: rgba(255, 255, 255, 0.22);
            box-shadow: none;
        }
        .panel-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
        }
        .panel h2 {
            margin: 0 0 4px;
            font-size: 1rem;
            letter-spacing: -0.03em;
        }
        .panel p {
            margin: 0;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }
        .step-badge {
            min-width: 34px;
            height: 34px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            border: 1px solid rgba(126, 117, 140, 0.08);
            background: rgba(255, 255, 255, 0.45);
            color: #746b7f;
            font-weight: 800;
        }
        .field-grid {
            display: grid;
            gap: 14px;
        }
        .field-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        @media (max-width: 760px) {
            .field-row,
            .hero-meta {
                grid-template-columns: 1fr;
            }
        }
        .field {
            display: grid;
            gap: 8px;
        }
        label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        input, select, textarea, button {
            width: 100%;
            border-radius: var(--radius-sm);
            border: 1px solid rgba(126, 117, 140, 0.08);
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.58);
            color: var(--ink);
            transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
            font: inherit;
        }
        input::placeholder,
        textarea::placeholder {
            color: rgba(185, 203, 230, 0.55);
        }
        select {
            appearance: none;
            -webkit-appearance: none;
            -moz-appearance: none;
        }
        select option {
            color: #2f2b38;
            background: #fffdfd;
        }
        select option:checked,
        select option:hover,
        select option:focus {
            color: #2f2b38;
            background: #fde7ef;
        }
        input:focus,
        select:focus,
        textarea:focus {
            outline: none;
            border-color: rgba(243, 169, 216, 0.45);
            box-shadow: 0 0 0 4px rgba(243, 169, 216, 0.14);
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        button {
            cursor: pointer;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            color: #47394a;
            font-weight: 800;
            border: none;
            letter-spacing: 0.01em;
            box-shadow: 0 12px 24px rgba(244, 161, 174, 0.16);
        }
        button:hover {
            filter: brightness(1.04);
            transform: translateY(-1px);
        }
        .ghost-button {
            background: rgba(255, 255, 255, 0.04);
            color: var(--ink);
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: none;
        }
        .sub {
            color: var(--muted);
            font-size: 12px;
        }
        .tip-list {
            display: none;
        }
        .tip {
            padding: 12px 14px;
            border-radius: var(--radius-sm);
            background: rgba(17, 31, 54, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .tip strong {
            display: block;
            color: var(--ink);
            margin-bottom: 4px;
        }
        .response-shell {
            padding: 0;
            overflow: hidden;
            position: static;
            background: rgba(255, 255, 255, 0.18);
            border: none;
            box-shadow: none;
            order: -1;
            grid-column: 1 / -1;
            max-width: 920px;
            width: 100%;
            justify-self: center;
        }
        .response-top {
            display: grid;
            justify-items: center;
            gap: 12px;
            padding: 8px 20px 4px;
            border-bottom: none;
            text-align: center;
        }
        .status-line {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.46);
            color: #72687c;
            border: 1px solid rgba(126, 117, 140, 0.08);
            font-weight: 700;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 0 6px rgba(243, 169, 216, 0.16);
        }
        .status-line.error {
            background: rgba(255, 222, 228, 0.8);
            color: #b25e72;
            border-color: rgba(255, 111, 134, 0.18);
        }
        .response-grid {
            display: grid;
            gap: 14px;
            padding: 12px 20px 22px;
        }
        .summary-grid {
            display: none;
        }
        .summary-card {
            padding: 14px 16px;
            border-radius: var(--radius-md);
            background: rgba(17, 32, 56, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .summary-card span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .summary-card strong {
            display: block;
            font-size: 1rem;
            line-height: 1.35;
        }
        pre {
            margin: 0;
            background: linear-gradient(180deg, rgba(10, 19, 35, 0.96), rgba(9, 17, 30, 0.98));
            border-radius: var(--radius-md);
            padding: 18px;
            max-height: 720px;
            overflow: auto;
            border: 1px solid rgba(255, 255, 255, 0.10);
            white-space: pre-wrap;
            line-height: 1.58;
            font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
            font-size: 0.92rem;
        }
        .helper-grid {
            display: grid;
            gap: 12px;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .chip {
            padding: 9px 12px;
            border-radius: 999px;
            border: 1px solid rgba(87, 189, 247, 0.16);
            background: rgba(87, 189, 247, 0.10);
            color: #cdeaff;
            font-size: 0.84rem;
            font-weight: 700;
        }
        .hidden {
            display: none !important;
        }
        .messages {
            display: grid;
            gap: 14px;
            max-height: 460px;
            overflow: auto;
            padding: 10px 4px 4px;
        }
        .bubble-row {
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        .bubble-row.user {
            justify-content: flex-end;
        }
        .avatar {
            width: 28px;
            height: 28px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            font-size: 0.72rem;
            font-weight: 800;
            background: rgba(255, 255, 255, 0.48);
            border: 1px solid rgba(126, 117, 140, 0.08);
            color: #746b7f;
            flex: 0 0 auto;
        }
        .bubble-row.user .avatar {
            order: 2;
            background: rgba(82, 216, 184, 0.12);
            border-color: rgba(82, 216, 184, 0.18);
            color: #d7fff4;
        }
        .bubble {
            max-width: min(100%, 760px);
            padding: 14px 16px;
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.32);
            box-shadow: 0 10px 24px rgba(120, 102, 132, 0.06);
        }
        .bubble.assistant {
            background: rgba(255, 255, 255, 0.48);
        }
        .bubble.user {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.58), rgba(255, 242, 246, 0.74));
        }
        .bubble-title {
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 8px;
        }
        .bubble-body {
            white-space: pre-wrap;
            line-height: 1.6;
        }
        .bubble-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        .bubble-chip {
            padding: 8px 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(126, 117, 140, 0.08);
            color: #746b7f;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .composer-shell {
            display: grid;
            gap: 10px;
        }
        .prompt-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }
        .prompt-chip {
            width: auto;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.44);
            color: #71667b;
            border: 1px solid rgba(126, 117, 140, 0.08);
            box-shadow: none;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .composer-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 10px;
            align-items: end;
            background: rgba(255, 255, 255, 0.48);
            border: 1px solid rgba(255, 255, 255, 0.36);
            border-radius: 999px;
            padding: 10px;
            box-shadow: 0 10px 30px rgba(120, 102, 132, 0.08);
        }
        .composer-row textarea {
            min-height: 56px;
            border: none;
            background: transparent;
            box-shadow: none;
            padding: 10px 12px;
        }
        @media (max-width: 760px) {
            .composer-row {
                grid-template-columns: 1fr;
                border-radius: 28px;
            }
        }
        .debug-block {
            display: grid;
            gap: 12px;
            opacity: 0.88;
        }
        #composer {
            resize: none;
        }
        label[for="composer"] {
            display: none;
        }
        .debug-block .ghost-button {
            width: auto;
            justify-self: center;
            padding-inline: 20px;
        }
        .debug-block pre {
            max-height: 180px;
            background: rgba(255, 255, 255, 0.34);
            border: 1px solid rgba(255, 255, 255, 0.36);
            color: #61586b;
        }
        .workspace > .panel {
            max-width: 100%;
            width: 100%;
            justify-self: stretch;
        }
        @media (max-width: 1180px) {
            .response-shell {
                position: static;
            }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <div>
                <div class="eyebrow">Patient Triage</div>
                <h1>Patient Triage OpenEnv</h1>
                <p>Describe symptoms. Get a triage direction.</p>
            </div>
            <div class="hero-meta">
                <div class="hero-stat">
                    <strong>3 tasks</strong>
                    <span>Easy, medium, and hard triage workflows</span>
                </div>
                <div class="hero-stat">
                    <strong>0.0 to 1.0</strong>
                    <span>Deterministic grading with shaped rewards</span>
                </div>
                <div class="hero-stat">
                    <strong>FastAPI</strong>
                    <span>Live reset / step / state inspection</span>
                </div>
            </div>
        </section>

        <section class="workspace">
            <div class="panel">
                <div class="panel-heading">
                    <div>
                        <h2>Session</h2>
                        <p>Task and seed.</p>
                    </div>
                    <div class="step-badge">1</div>
                </div>
                <div class="field-grid">
                    <div class="field">
                        <label for="task">Task</label>
                        <select id="task">
                            <option value="urgency_classification">urgency_classification</option>
                            <option value="care_recommendation">care_recommendation</option>
                            <option value="full_triage_decision" selected>full_triage_decision</option>
                        </select>
                    </div>
                    <div class="field">
                        <label for="seed">Seed</label>
                        <input id="seed" type="number" value="0" />
                    </div>
                    <button onclick="resetEpisode()">Reset Episode</button>
                </div>
                <div class="tip-list">
                    <div class="tip">
                        <strong>Good workflow</strong>
                        Reset first, inspect the case summary, then submit only the action fields needed for the current action type.
                    </div>
                    <div class="tip">
                        <strong>Deterministic demos</strong>
                        Keep the same seed when you want to reproduce a strong case for judges or teammates.
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-heading">
                    <div>
                        <h2>Advanced</h2>
                        <p>Structured action.</p>
                    </div>
                    <div class="step-badge">2</div>
                </div>
                <div class="field-grid">
                    <div class="field">
                        <label for="action_type">Action Type</label>
                        <select id="action_type" onchange="syncActionFields()">
                            <option value="ask_question">ask_question</option>
                            <option value="assign_urgency">assign_urgency</option>
                            <option value="recommend_care">recommend_care</option>
                            <option value="finalize" selected>finalize</option>
                        </select>
                    </div>
                    <div class="field" id="question_id_wrap">
                        <label for="question_id">Question ID</label>
                        <input id="question_id" placeholder="Ask one available clarifying question" />
                    </div>
                    <div class="field-row">
                        <div class="field" id="urgency_wrap">
                            <label for="urgency">Urgency</label>
                            <select id="urgency">
                                <option value="">(none)</option>
                                <option value="low">low</option>
                                <option value="medium">medium</option>
                                <option value="high">high</option>
                                <option value="critical">critical</option>
                            </select>
                        </div>
                        <div class="field" id="care_destination_wrap">
                            <label for="care_destination">Care Destination</label>
                            <select id="care_destination">
                                <option value="">(none)</option>
                                <option value="home_care">home_care</option>
                                <option value="clinic">clinic</option>
                                <option value="urgent_care">urgent_care</option>
                                <option value="emergency_room">emergency_room</option>
                            </select>
                        </div>
                    </div>
                    <div class="field" id="reason_codes_wrap">
                        <label for="reason_codes">Reason codes</label>
                        <input id="reason_codes" placeholder="red_flag_symptom,abnormal_vitals" />
                    </div>
                    <div class="field" id="rationale_wrap">
                        <label for="rationale">Rationale</label>
                        <textarea id="rationale" placeholder="Concise clinical reasoning"></textarea>
                    </div>
                    <button onclick="sendAdvancedAction()">Submit Step</button>
                </div>
            </div>

            <section class="response-shell">
                <div class="response-top">
                    <div>
                        <h2 style="margin:0 0 4px;">Ask anything</h2>
                        <p style="margin:0;color:var(--muted);font-size:.92rem;">Symptoms, disease, or a command.</p>
                    </div>
                    <div class="status-line" id="status_pill">
                        <span class="status-dot"></span>
                        <span id="status_text">Ready for your first message.</span>
                    </div>
                </div>
                <div class="response-grid">
                    <div class="summary-grid">
                        <div class="summary-card">
                            <span>Task</span>
                            <strong id="summary_task">Not started</strong>
                        </div>
                        <div class="summary-card">
                            <span>Case</span>
                            <strong id="summary_case">No active case</strong>
                        </div>
                        <div class="summary-card">
                            <span>Progress</span>
                            <strong id="summary_progress">0 steps - pending</strong>
                        </div>
                    </div>

                    <div class="messages" id="messages"></div>

                    <div class="composer-shell">
                        <div class="prompt-grid">
                            <button class="prompt-chip" onclick="fillPrompt('Chest pain with sweating and shortness of breath')">Chest pain</button>
                            <button class="prompt-chip" onclick="fillPrompt('Child with high fever and low energy')">High fever</button>
                            <button class="prompt-chip" onclick="fillPrompt('Severe allergic reaction with throat swelling')">Allergic reaction</button>
                            <button class="prompt-chip" onclick="fillPrompt('/state')">/state</button>
                        </div>
                        <div class="composer-row">
                            <div class="field">
                                <label for="composer">Message</label>
                                <textarea id="composer" placeholder="Describe a disease or symptoms, or use /reset, /ask, /urgency, /care, /finalize"></textarea>
                            </div>
                            <button onclick="handleComposer()">Send</button>
                        </div>
                    </div>

                    <div class="debug-block">
                        <button class="ghost-button" onclick="getState()">Refresh State</button>
                        <pre id="output">Ready.

Tip: Start with Reset Episode to load a patient case, or type a symptom description to get a demo triage suggestion.</pre>
                    </div>
                </div>
            </section>
        </section>
    </div>

    <script>
        const messages = document.getElementById("messages");
        const output = document.getElementById("output");
        const statusPill = document.getElementById("status_pill");
        const statusText = document.getElementById("status_text");
        const summaryTask = document.getElementById("summary_task");
        const summaryCase = document.getElementById("summary_case");
        const summaryProgress = document.getElementById("summary_progress");
        const composer = document.getElementById("composer");

        function escapeHtml(value) {
            return String(value)
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;");
        }

        function setStatus(msg, isError = false) {
            statusPill.classList.toggle("error", isError);
            statusText.textContent = msg;
        }

        function addMessage(role, title, body, chips = []) {
            const row = document.createElement("div");
            row.className = `bubble-row ${role}`;
            const meta = chips.length
                ? `<div class="bubble-meta">${chips.map((item) => `<span class="bubble-chip">${escapeHtml(item)}</span>`).join("")}</div>`
                : "";
            row.innerHTML = `
                <div class="avatar">${role === "assistant" ? "AI" : "You"}</div>
                <div class="bubble ${role}">
                    <div class="bubble-title">${escapeHtml(title)}</div>
                    <div class="bubble-body">${escapeHtml(body)}</div>
                    ${meta}
                </div>
            `;
            messages.appendChild(row);
            messages.scrollTop = messages.scrollHeight;
        }

        function fillPrompt(value) {
            composer.value = value;
            composer.focus();
        }

        function welcomeMessage() {
            addMessage(
                "assistant",
                "Welcome",
                "Describe a disease or symptom pattern and I will return a benchmark-style triage suggestion. If you want to operate the actual environment, use commands like /reset, /ask, /urgency, /care, /finalize, or /state.",
                ["free-text demo", "real env commands", "deterministic output"]
            );
        }

        function showJson(data) {
            output.textContent = JSON.stringify(data, null, 2);
        }

        function updateSummary(data) {
            const observation = data.observation || {};
            const state = data.state || {};

            summaryTask.textContent = observation.task_name || state.task_name || "Not started";
            summaryCase.textContent = observation.case_id || state.case_id || "No active case";

            if (state.step_count !== undefined) {
                const finalScore = state.final_score !== undefined && state.final_score !== null
                    ? `final score ${Number(state.final_score).toFixed(2)}`
                    : "active";
                summaryProgress.textContent = `${state.step_count} steps - ${finalScore}`;
            } else if (observation.done) {
                summaryProgress.textContent = "Episode complete";
            } else if (observation.remaining_steps !== undefined) {
                summaryProgress.textContent = `${observation.remaining_steps} steps remaining`;
            } else {
                summaryProgress.textContent = "0 steps - pending";
            }
        }

        function syncActionFields() {
            const actionType = document.getElementById("action_type").value;
            const questionOnly = actionType === "ask_question";
            const urgencyOnly = actionType === "assign_urgency";
            const careOnly = actionType === "recommend_care";
            const finalizing = actionType === "finalize";

            document.getElementById("question_id_wrap").classList.toggle("hidden", !questionOnly);
            document.getElementById("urgency_wrap").classList.toggle("hidden", !(urgencyOnly || finalizing));
            document.getElementById("care_destination_wrap").classList.toggle("hidden", !(careOnly || finalizing));
            document.getElementById("reason_codes_wrap").classList.toggle("hidden", questionOnly);
            document.getElementById("rationale_wrap").classList.toggle("hidden", questionOnly);
        }

        function advancedActionPayload() {
            const action = {
                action_type: document.getElementById("action_type").value,
            };
            const questionId = document.getElementById("question_id").value.trim();
            const urgency = document.getElementById("urgency").value;
            const careDestination = document.getElementById("care_destination").value;
            const reasonCodesRaw = document.getElementById("reason_codes").value.trim();
            const rationale = document.getElementById("rationale").value.trim();

            if (questionId) action.question_id = questionId;
            if (urgency) action.urgency = urgency;
            if (careDestination) action.care_destination = careDestination;
            if (reasonCodesRaw) action.reason_codes = reasonCodesRaw.split(",").map((s) => s.trim()).filter(Boolean);
            if (rationale) action.rationale = rationale;
            return action;
        }

        function fillAdvancedAction(action) {
            if (!action) return;
            document.getElementById("action_type").value = action.action_type || "finalize";
            document.getElementById("question_id").value = action.question_id || "";
            document.getElementById("urgency").value = action.urgency || "";
            document.getElementById("care_destination").value = action.care_destination || "";
            document.getElementById("reason_codes").value = (action.reason_codes || []).join(",");
            document.getElementById("rationale").value = action.rationale || "";
            syncActionFields();
        }

        function observationText(data) {
            const observation = data.observation || {};
            const provisional = observation.provisional_decision || {};
            const knownAnswers = observation.known_answers || {};
            const answerLines = Object.keys(knownAnswers).length
                ? Object.entries(knownAnswers).map(([k, v]) => `- ${k}: ${v}`).join("\\n")
                : "none yet";

            return [
                observation.patient_summary ? `Patient summary:\\n${observation.patient_summary}` : null,
                observation.feedback ? `Feedback:\\n${observation.feedback}` : null,
                `Provisional urgency: ${provisional.urgency || "not set"}`,
                `Provisional care: ${provisional.care_destination || "not set"}`,
                `Reason codes: ${(provisional.reason_codes || []).join(", ") || "none"}`,
                `Known answers:\\n${answerLines}`,
            ].filter(Boolean).join("\\n\\n");
        }

        function actionSummary(action) {
            const parts = [`type=${action.action_type}`];
            if (action.question_id) parts.push(`question=${action.question_id}`);
            if (action.urgency) parts.push(`urgency=${action.urgency}`);
            if (action.care_destination) parts.push(`care=${action.care_destination}`);
            if (action.reason_codes && action.reason_codes.length) parts.push(`reasons=${action.reason_codes.join(",")}`);
            if (action.rationale) parts.push(`rationale=${action.rationale}`);
            return parts.join(" | ");
        }

        function commandActionFromText(text) {
            const trimmed = text.trim();
            const lower = trimmed.toLowerCase();
            if (lower.startsWith("/ask ")) {
                return { action_type: "ask_question", question_id: trimmed.slice(5).trim() };
            }
            if (lower.startsWith("/urgency ")) {
                const payload = trimmed.slice(9).trim();
                const [urgency, ...tail] = payload.split(/\\s+/);
                return { action_type: "assign_urgency", urgency, rationale: tail.join(" ") || undefined };
            }
            if (lower.startsWith("/care ")) {
                const payload = trimmed.slice(6).trim();
                const [careDestination, ...tail] = payload.split(/\\s+/);
                return { action_type: "recommend_care", care_destination: careDestination, rationale: tail.join(" ") || undefined };
            }
            if (lower.startsWith("/finalize")) {
                const payload = trimmed.slice(9).trim();
                const action = { action_type: "finalize" };
                const pairs = Array.from(payload.matchAll(/(urgency|care|reasons|rationale)=(".*?"|\\S+)/g));
                pairs.forEach((match) => {
                    const key = match[1];
                    const value = match[2].replace(/^"|"$/g, "");
                    if (key === "urgency") action.urgency = value;
                    if (key === "care") action.care_destination = value;
                    if (key === "reasons") action.reason_codes = value.split(",").map((item) => item.trim()).filter(Boolean);
                    if (key === "rationale") action.rationale = value;
                });
                if (!pairs.length && payload) {
                    action.rationale = payload;
                }
                return action;
            }
            return null;
        }

        async function callApi(path, method, body) {
            const res = await fetch(path, {
                method,
                headers: { "Content-Type": "application/json" },
                body: body ? JSON.stringify(body) : undefined,
            });
            const data = await res.json().catch(() => ({ detail: "Non-JSON response" }));
            if (!res.ok) {
                throw new Error(data.detail || `HTTP ${res.status}`);
            }
            return data;
        }

        async function resetEpisode() {
            try {
                setStatus("Resetting episode...");
                const task = document.getElementById("task").value;
                const seed = Number(document.getElementById("seed").value || 0);
                const data = await callApi("/reset", "POST", { task, seed });
                showJson(data);
                updateSummary(data);
                addMessage(
                    "assistant",
                    "Episode Reset",
                    observationText(data),
                    [data.observation.task_name, data.observation.case_id, `remaining ${data.observation.remaining_steps}`]
                );
                setStatus("Episode reset complete.");
            } catch (err) {
                setStatus(err.message, true);
                addMessage("assistant", "Reset failed", String(err.message), ["error"]);
            }
        }

        async function stepEpisode(action, label = null) {
            try {
                setStatus("Submitting action...");
                const data = await callApi("/step", "POST", { action });
                showJson(data);
                updateSummary(data);
                addMessage("user", "Action", label || actionSummary(action));
                addMessage(
                    "assistant",
                    data.done ? "Episode Result" : "Environment Response",
                    observationText(data),
                    [data.done ? "done" : `remaining ${data.observation.remaining_steps}`, `reward ${Number(data.reward).toFixed(2)}`]
                );
                setStatus("Action submitted.");
            } catch (err) {
                setStatus(err.message, true);
                addMessage("assistant", "Step failed", String(err.message), ["error"]);
            }
        }

        async function getState() {
            try {
                setStatus("Fetching state...");
                const data = await callApi("/state", "GET");
                showJson(data);
                updateSummary(data);
                const state = data.state || {};
                addMessage(
                    "assistant",
                    "Current State",
                    `Task: ${state.task_name || "n/a"}\\nCase: ${state.case_id || "n/a"}\\nStep count: ${state.step_count ?? 0}\\nBest partial score: ${state.best_partial_score ?? 0}\\nFinal score: ${state.final_score ?? "not set"}\\nLast feedback: ${state.last_feedback || "n/a"}`,
                    [state.difficulty || "no difficulty", `${state.remaining_steps ?? 0} remaining`]
                );
                setStatus("State fetched.");
            } catch (err) {
                setStatus(err.message, true);
                addMessage("assistant", "State request failed", String(err.message), ["error"]);
            }
        }

        async function sendAdvancedAction() {
            const action = advancedActionPayload();
            await stepEpisode(action);
        }

        async function handleDemoPrompt(text) {
            try {
                setStatus("Generating triage suggestion...");
                const data = await callApi("/assistant/triage", "POST", { message: text });
                fillAdvancedAction(data.suggested_action);
                addMessage(
                    "assistant",
                    "Triage Suggestion",
                    `${data.answer}\\n\\nReasoning:\\n${data.rationale}\\n\\n${data.disclaimer}`,
                    [data.urgency, data.care_destination, `${data.reason_codes.length} reason codes`]
                );
                setStatus("Suggestion ready.");
            } catch (err) {
                setStatus(err.message, true);
                addMessage("assistant", "Suggestion failed", String(err.message), ["error"]);
            }
        }

        async function handleComposer() {
            const text = composer.value.trim();
            if (!text) return;

            addMessage("user", "You", text);
            composer.value = "";

            const lower = text.toLowerCase();
            if (lower.startsWith("/reset")) {
                const [, task, seed] = text.split(/\s+/);
                if (task) document.getElementById("task").value = task;
                if (seed !== undefined) document.getElementById("seed").value = seed;
                await resetEpisode();
                return;
            }
            if (lower.startsWith("/state")) {
                await getState();
                return;
            }

            const action = commandActionFromText(text);
            if (action) {
                fillAdvancedAction(action);
                await stepEpisode(action);
                return;
            }

            await handleDemoPrompt(text);
        }

        composer.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleComposer();
            }
        });

        syncActionFields();
        welcomeMessage();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.get("/status")
def status() -> dict[str, object]:
    return {
        "name": "patient-triage-openenv",
        "status": "running",
        "endpoints": ["/", "/status", "/healthz", "/reset", "/step", "/state", "/ws", "/docs"],
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/assistant/triage")
def assistant_triage(payload: dict = Body(default_factory=dict)) -> dict[str, object]:
    message = str(payload.get("message", "")).strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    try:
        return _chat_triage_suggestion(message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> dict:
    try:
        return env.reset(request).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(request: StepRequest) -> dict:
    try:
        return env.step(request.action).model_dump(mode="json")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict:
    try:
        return StateEnvelope(state=env.state()).model_dump(mode="json")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            payload = message.get("data", {})
            if message_type == "reset":
                try:
                    response = env.reset(ResetRequest.model_validate(payload))
                except (RuntimeError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "data": {"code": "BAD_REQUEST", "message": str(exc)}})
                    continue
                await websocket.send_json({"type": "reset_result", "data": response.model_dump(mode="json")})
            elif message_type == "step":
                try:
                    response = env.step(StepRequest.model_validate({"action": payload}).action)
                except (RuntimeError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "data": {"code": "BAD_REQUEST", "message": str(exc)}})
                    continue
                await websocket.send_json({"type": "step_result", "data": response.model_dump(mode="json")})
            elif message_type == "state":
                try:
                    response = StateEnvelope(state=env.state())
                except RuntimeError as exc:
                    await websocket.send_json({"type": "error", "data": {"code": "BAD_REQUEST", "message": str(exc)}})
                    continue
                await websocket.send_json({"type": "state_result", "data": response.model_dump(mode="json")})
            elif message_type == "close":
                await websocket.send_json({"type": "closed", "data": {"status": "bye"}})
                await websocket.close()
                return
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"code": "BAD_MESSAGE", "message": f"Unsupported message type: {message_type}"},
                    }
                )
    except WebSocketDisconnect:
        return
