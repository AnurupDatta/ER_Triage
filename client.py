# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ER Triage Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import ERTriageAction, ERTriageObservation, ERTriageState
except ImportError:
    from models import ERTriageAction, ERTriageObservation, ERTriageState


class ERTriageEnv(
    EnvClient[ERTriageAction, ERTriageObservation, ERTriageState]
):
    """
    Client for the ER Triage Environment.

    Example:
        >>> with ERTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.chief_complaint)
        ...
        ...     action = ERTriageAction(action_type="request_vitals", reasoning="Initial assessment")
        ...     result = client.step(action)
        ...     print(result.observation.vitals)
    """

    def _step_payload(self, action: ERTriageAction) -> Dict[str, Any]:
        payload = {
            "action_type": action.action_type,
            "reasoning": action.reasoning,
        }
        if action.action_type == "ask_question":
            payload["question"] = action.question
        elif action.action_type == "assign_priority":
            payload["priority"] = action.priority
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ERTriageObservation]:
        obs_data = payload.get("observation", {})
        observation = ERTriageObservation(
            patient_id=obs_data.get("patient_id", ""),
            chief_complaint=obs_data.get("chief_complaint", ""),
            vitals=obs_data.get("vitals"),
            history=obs_data.get("history"),
            question_answer=obs_data.get("question_answer"),
            available_actions=obs_data.get("available_actions", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ERTriageState:
        return ERTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "Unknown"),
            patient_queue=payload.get("patient_queue", []),
            current_patient_index=payload.get("current_patient_index", 0),
            steps_taken_for_patient=payload.get("steps_taken_for_patient", 0),
            bias_log=payload.get("bias_log", []),
        )
