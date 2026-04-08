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
            --bg: #0b1320;
            --card: #10213a;
            --card-2: #132a48;
            --ink: #eaf2ff;
            --muted: #9bb2d3;
            --accent: #37c6a8;
            --warn: #ff8f6b;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", "Trebuchet MS", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(1200px 700px at -10% -10%, #1f446f 0%, transparent 45%),
                radial-gradient(900px 500px at 110% 0%, #1a6b5f 0%, transparent 40%),
                var(--bg);
            min-height: 100vh;
            padding: 20px;
        }
        .wrap {
            max-width: 980px;
            margin: 0 auto;
            display: grid;
            gap: 16px;
        }
        .hero, .panel {
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 16px;
            background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.01));
            backdrop-filter: blur(4px);
            box-shadow: 0 12px 30px rgba(0,0,0,.28);
        }
        .hero { padding: 20px; }
        .hero h1 { margin: 0 0 8px; font-size: clamp(22px, 3vw, 34px); letter-spacing: .3px; }
        .hero p { margin: 0; color: var(--muted); }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        @media (max-width: 840px) { .row { grid-template-columns: 1fr; } }
        .panel { padding: 16px; }
        .panel h2 { margin: 0 0 10px; font-size: 18px; }
        .grid { display: grid; gap: 8px; }
        label { color: var(--muted); font-size: 13px; }
        input, select, textarea, button {
            width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,.18);
            padding: 10px; background: var(--card-2); color: var(--ink);
        }
        textarea { min-height: 80px; resize: vertical; }
        button {
            cursor: pointer; background: linear-gradient(90deg, var(--accent), #4fd0ff);
            color: #07222d; font-weight: 700; border: none;
        }
        button:hover { filter: brightness(1.05); }
        .sub { color: var(--muted); font-size: 12px; }
        pre {
            margin: 0; background: var(--card); border-radius: 10px; padding: 12px;
            max-height: 360px; overflow: auto; border: 1px solid rgba(255,255,255,.1);
            white-space: pre-wrap;
        }
        .status { color: var(--warn); font-weight: 600; min-height: 20px; }
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <h1>Patient Triage OpenEnv</h1>
            <p>Interactive frontend for the triage benchmark API. Use reset then submit actions through step.</p>
        </section>

        <section class="row">
            <div class="panel grid">
                <h2>1) Reset Episode</h2>
                <label>Task</label>
                <select id="task">
                    <option value="urgency_classification">urgency_classification</option>
                    <option value="care_recommendation">care_recommendation</option>
                    <option value="full_triage_decision" selected>full_triage_decision</option>
                </select>
                <label>Seed</label>
                <input id="seed" type="number" value="0" />
                <button onclick="resetEpisode()">Reset</button>
            </div>

            <div class="panel grid">
                <h2>2) Submit Action</h2>
                <label>Action Type</label>
                <select id="action_type">
                    <option value="ask_question">ask_question</option>
                    <option value="assign_urgency">assign_urgency</option>
                    <option value="recommend_care">recommend_care</option>
                    <option value="finalize" selected>finalize</option>
                </select>
                <label>Question ID</label>
                <input id="question_id" placeholder="optional" />
                <label>Urgency</label>
                <select id="urgency">
                    <option value="">(none)</option>
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                    <option value="critical">critical</option>
                </select>
                <label>Care Destination</label>
                <select id="care_destination">
                    <option value="">(none)</option>
                    <option value="home_care">home_care</option>
                    <option value="clinic">clinic</option>
                    <option value="urgent_care">urgent_care</option>
                    <option value="emergency_room">emergency_room</option>
                </select>
                <label>Reason codes (comma separated)</label>
                <input id="reason_codes" placeholder="red_flag_symptom,abnormal_vitals" />
                <label>Rationale</label>
                <textarea id="rationale" placeholder="Short reasoning"></textarea>
                <button onclick="stepEpisode()">Step</button>
            </div>
        </section>

        <section class="panel grid">
            <h2>3) Current State</h2>
            <div class="sub">Use this after reset/step to inspect current environment state.</div>
            <button onclick="getState()">Get State</button>
            <div class="status" id="status"></div>
            <pre id="output">Ready.</pre>
        </section>
    </div>

    <script>
        const output = document.getElementById("output");
        const statusEl = document.getElementById("status");

        function setStatus(msg, isError = false) {
            statusEl.style.color = isError ? "#ff8f6b" : "#37c6a8";
            statusEl.textContent = msg;
        }

        function showJson(data) {
            output.textContent = JSON.stringify(data, null, 2);
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
                setStatus("State fetched.");
            } catch (err) {
                setStatus(err.message, true);
            }
        }
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
