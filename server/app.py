"""Root server entrypoint expected by OpenEnv validation."""

from __future__ import annotations

import os

import uvicorn

from patient_triage_env.server.app import app


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
