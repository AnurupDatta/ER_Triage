# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Er Triage 1 Environment.

This module creates an HTTP server that exposes the ErTriage1Environment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ERTriageAction, ERTriageObservation
    from .er_triage_environment import ERTriageEnvironment
except (ImportError, ModuleNotFoundError):
    from models import ERTriageAction, ERTriageObservation
    from server.er_triage_environment import ERTriageEnvironment


app = create_app(
    ERTriageEnvironment,
    ERTriageAction,
    ERTriageObservation,
    env_name="er_triage",
    max_concurrent_envs=10,
)


# ── Task & Grader Definitions ────────────────────────────────────────────────

TASKS = [
    {
        "id": "single_triage",
        "description": "Triage 1 patient using ESI protocol",
        "difficulty": "easy",
        "max_attempts": 4,
        "scoring": "partial credit strictly within (0, 1)",
        "action_schema": ERTriageAction.model_json_schema(),
    },
    {
        "id": "batch_triage",
        "description": "Triage 3 patients sequentially",
        "difficulty": "medium",
        "max_attempts": 12,
        "scoring": "partial credit strictly within (0, 1)",
        "action_schema": ERTriageAction.model_json_schema(),
    },
    {
        "id": "differential_triage",
        "description": "Triage 1 tricky patient with misleading symptoms",
        "difficulty": "hard",
        "max_attempts": 4,
        "scoring": "partial credit strictly within (0, 1)",
        "action_schema": ERTriageAction.model_json_schema(),
    },
]


@app.get("/tasks", tags=["Tasks"])
def list_tasks():
    """List all tasks with their action schema."""
    return TASKS


@app.post("/grader", tags=["Grader"])
def grade(payload: dict):
    """Grade a priority assignment for a given patient.

    Expects: {"task_id": str, "patient_id": str, "priority": str}
    Returns: {"score": float, "correct": bool, "ground_truth": str}
    """
    try:
        from data.patients import PATIENTS
    except ImportError:
        from ..data.patients import PATIENTS

    eps = 0.001
    task_id = payload.get("task_id", "single_triage")
    patient_id = payload.get("patient_id")
    assigned_priority = payload.get("priority")

    if not patient_id or not assigned_priority:
        return {"error": "patient_id and priority are required", "score": eps}

    patient = next((p for p in PATIENTS if p["patient_id"] == patient_id), None)
    if not patient:
        return {"error": f"Unknown patient_id: {patient_id}", "score": eps}

    ground_truth = patient["ground_truth_priority"]
    correct = assigned_priority == ground_truth

    # Partial credit based on priority distance
    _levels = ["non-urgent", "urgent", "critical"]
    _idx = {p: i for i, p in enumerate(_levels)}
    if correct:
        raw = 0.70
    else:
        distance = abs(_idx.get(ground_truth, 1) - _idx.get(assigned_priority, 1))
        raw = 0.30 if distance == 1 else 0.05

    eps = 0.001
    score = min(max(raw, eps), 1.0 - eps)

    return {
        "task_id": task_id,
        "patient_id": patient_id,
        "score": score,
        "correct": correct,
        "ground_truth": ground_truth,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m er_triage.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn er_triage.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
