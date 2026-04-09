"""FastAPI app exposing reset, step, state, and WebSocket APIs."""

from __future__ import annotations

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
            --bg: #07111d;
            --panel: rgba(13, 24, 42, 0.82);
            --panel-strong: rgba(18, 34, 58, 0.92);
            --panel-soft: #142b49;
            --border: rgba(165, 206, 255, 0.14);
            --border-strong: rgba(110, 200, 185, 0.28);
            --ink: #edf4ff;
            --muted: #95abc8;
            --accent: #52d8b8;
            --accent-2: #57bdf7;
            --warn: #ff8f6b;
            --danger: #ff6f86;
            --shadow: 0 28px 64px rgba(1, 7, 17, 0.42);
            --radius-xl: 28px;
            --radius-lg: 22px;
            --radius-md: 16px;
            --radius-sm: 12px;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Aptos, "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(1100px 720px at -8% -12%, rgba(48, 97, 156, 0.55) 0%, transparent 45%),
                radial-gradient(920px 560px at 108% -4%, rgba(27, 126, 109, 0.46) 0%, transparent 42%),
                radial-gradient(680px 420px at 50% 120%, rgba(42, 104, 180, 0.18) 0%, transparent 60%),
                var(--bg);
            min-height: 100vh;
            padding: 28px 20px 52px;
        }
        .wrap {
            max-width: 1320px;
            margin: 0 auto;
            display: grid;
            gap: 22px;
        }
        .hero,
        .panel,
        .response-shell {
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            background:
                linear-gradient(160deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015)),
                var(--panel);
            backdrop-filter: blur(16px);
            box-shadow: var(--shadow);
        }
        .hero {
            padding: 28px;
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.9fr);
            gap: 24px;
            align-items: end;
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: auto -120px -140px auto;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(87, 189, 247, 0.20), transparent 68%);
            pointer-events: none;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(82, 216, 184, 0.12);
            color: #b7f7e6;
            border: 1px solid rgba(82, 216, 184, 0.18);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .hero h1 {
            margin: 14px 0 12px;
            font-size: clamp(2.2rem, 5vw, 4rem);
            line-height: 0.98;
            letter-spacing: -0.05em;
        }
        .hero p {
            margin: 0;
            max-width: 760px;
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.65;
        }
        .hero-meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
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
            grid-template-columns: minmax(340px, 0.88fr) minmax(360px, 0.98fr) minmax(320px, 1.14fr);
            gap: 22px;
            align-items: start;
        }
        @media (max-width: 1180px) {
            .hero,
            .workspace {
                grid-template-columns: 1fr;
            }
        }
        .panel {
            padding: 22px 20px 20px;
        }
        .panel-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 18px;
        }
        .panel h2 {
            margin: 0 0 4px;
            font-size: 1.32rem;
            letter-spacing: -0.03em;
        }
        .panel p {
            margin: 0;
            color: var(--muted);
            font-size: 0.93rem;
            line-height: 1.55;
        }
        .step-badge {
            min-width: 42px;
            height: 42px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            border: 1px solid rgba(87, 189, 247, 0.22);
            background: rgba(87, 189, 247, 0.12);
            color: #bde7ff;
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
            border: 1px solid rgba(121, 173, 236, 0.18);
            padding: 14px 16px;
            background: linear-gradient(180deg, rgba(28, 56, 94, 0.95), rgba(24, 48, 81, 0.95));
            color: var(--ink);
            transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
            font: inherit;
        }
        input::placeholder,
        textarea::placeholder {
            color: rgba(185, 203, 230, 0.55);
        }
        input:focus,
        select:focus,
        textarea:focus {
            outline: none;
            border-color: rgba(87, 189, 247, 0.55);
            box-shadow: 0 0 0 4px rgba(87, 189, 247, 0.12);
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        button {
            cursor: pointer;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            color: #07222d;
            font-weight: 800;
            border: none;
            letter-spacing: 0.01em;
            box-shadow: 0 16px 30px rgba(55, 198, 168, 0.18);
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
            display: grid;
            gap: 10px;
            margin-top: 18px;
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
            position: sticky;
            top: 20px;
        }
        .response-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            padding: 18px 20px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }
        .status-line {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 999px;
            background: rgba(82, 216, 184, 0.10);
            color: #c2fff0;
            border: 1px solid rgba(82, 216, 184, 0.14);
            font-weight: 700;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 0 6px rgba(82, 216, 184, 0.14);
        }
        .status-line.error {
            background: rgba(255, 111, 134, 0.12);
            color: #ffc0ca;
            border-color: rgba(255, 111, 134, 0.18);
        }
        .response-grid {
            display: grid;
            gap: 16px;
            padding: 18px 20px 22px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        @media (max-width: 760px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
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
                <div class="eyebrow">Clinical Reasoning Benchmark</div>
                <h1>Patient Triage OpenEnv</h1>
                <p>
                    A polished control room for your synthetic triage benchmark. Reset a case, compose an action with the
                    right fields, inspect the live environment response, and iterate safely without digging through raw API calls.
                </p>
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
                        <h2>1) Reset Episode</h2>
                        <p>Choose the benchmark task, lock a seed, and start a deterministic case.</p>
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
                        <h2>2) Compose Action</h2>
                        <p>The form adapts to the chosen action, so you only see relevant controls when you need them.</p>
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
                                <option value="clinic_visit">clinic_visit</option>
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
                    <button onclick="stepEpisode()">Submit Step</button>
                </div>
            </div>

            <section class="response-shell">
                <div class="response-top">
                    <div>
                        <h2 style="margin:0 0 4px;">3) Live Response Console</h2>
                        <p style="margin:0;color:var(--muted);font-size:.92rem;">Inspect the latest observation, reward, and state without leaving the page.</p>
                    </div>
                    <div class="status-line" id="status_pill">
                        <span class="status-dot"></span>
                        <span id="status_text">Ready for your first reset.</span>
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
                    <div class="helper-grid">
                        <div class="chip-row" id="summary_chips">
                            <div class="chip">reset -> step -> inspect</div>
                        </div>
                        <button class="ghost-button" onclick="getState()">Refresh State</button>
                    </div>
                    <pre id="output">Ready.

Tip: Start with Reset Episode to load a patient case. The full JSON response will appear here.</pre>
                </div>
            </section>
        </section>
    </div>

    <script>
        const output = document.getElementById("output");
        const statusPill = document.getElementById("status_pill");
        const statusText = document.getElementById("status_text");
        const summaryTask = document.getElementById("summary_task");
        const summaryCase = document.getElementById("summary_case");
        const summaryProgress = document.getElementById("summary_progress");
        const summaryChips = document.getElementById("summary_chips");

        function setStatus(msg, isError = false) {
            statusPill.classList.toggle("error", isError);
            statusText.textContent = msg;
        }

        function showJson(data) {
            output.textContent = JSON.stringify(data, null, 2);
        }

        function chipsForData(data) {
            const chips = [];
            const observation = data.observation || {};
            const state = data.state || {};
            const metadata = observation.metadata || {};
            const reasonCodeOptions = metadata.reason_code_options || [];

            if (observation.difficulty) chips.push(`difficulty: ${observation.difficulty}`);
            if (typeof observation.remaining_steps === "number") chips.push(`remaining: ${observation.remaining_steps}`);
            if (typeof observation.reward === "number") chips.push(`reward: ${observation.reward.toFixed(2)}`);
            if (state.final_score !== undefined && state.final_score !== null) chips.push(`final score: ${Number(state.final_score).toFixed(2)}`);
            if (reasonCodeOptions.length) chips.push(`${reasonCodeOptions.length} reason codes`);

            return chips.length ? chips : ["waiting for data"];
        }

        function renderChips(chips) {
            summaryChips.innerHTML = "";
            chips.forEach((chip) => {
                const node = document.createElement("div");
                node.className = "chip";
                node.textContent = chip;
                summaryChips.appendChild(node);
            });
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

            renderChips(chipsForData(data));
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
                setStatus("Episode reset complete.");
            } catch (err) {
                setStatus(err.message, true);
            }
        }

        async function stepEpisode() {
            try {
                setStatus("Submitting action...");
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
                if (reasonCodesRaw) action.reason_codes = reasonCodesRaw.split(",").map(s => s.trim()).filter(Boolean);
                if (rationale) action.rationale = rationale;

                const data = await callApi("/step", "POST", { action });
                showJson(data);
                updateSummary(data);
                setStatus("Action submitted.");
            } catch (err) {
                setStatus(err.message, true);
            }
        }

        async function getState() {
            try {
                setStatus("Fetching state...");
                const data = await callApi("/state", "GET");
                showJson(data);
                updateSummary(data);
                setStatus("State fetched.");
            } catch (err) {
                setStatus(err.message, true);
            }
        }

        syncActionFields();
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
