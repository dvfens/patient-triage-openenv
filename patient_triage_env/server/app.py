"""FastAPI app exposing reset, step, state, and WebSocket APIs."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from ..models import ResetRequest, StateEnvelope, StepRequest
from .environment import PatientTriageEnvironment


app = FastAPI(
    title="Patient Triage OpenEnv",
    version="0.1.0",
    description="Synthetic patient triage benchmark for agent evaluation.",
)
env = PatientTriageEnvironment()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "patient-triage-openenv",
        "status": "running",
        "endpoints": ["/healthz", "/reset", "/step", "/state", "/ws", "/docs"],
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest) -> dict:
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
