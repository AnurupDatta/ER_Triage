# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the ER Triage Environment."""

from typing import List, Dict, Optional, Literal
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ERTriageAction(Action):
    """
    Action for the ER Triage environment.
    The agent can request vitals, ask a clarifying question, or assign a priority.
    """
    action_type: Literal["request_vitals", "ask_question", "assign_priority"] = Field(
        ..., description="The type of action to perform."
    )
    question: Optional[str] = Field(
        None, description="The question to ask the patient (if action_type is 'ask_question')."
    )
    priority: Optional[Literal["critical", "urgent", "non-urgent"]] = Field(
        None, description="The ESI priority level to assign (if action_type is 'assign_priority')."
    )
    reasoning: str = Field(
        "", description="The agent's reasoning for taking this action."
    )


class ERTriageObservation(Observation):
    """
    Observation from the ER Triage environment.
    Provides the agent with the current patient's information.
    """
    patient_id: str
    chief_complaint: str
    vitals: Optional[Dict[str, float | str]] = None
    history: Optional[str] = None
    question_answer: Optional[str] = None
    available_actions: List[str] = Field(
        default_factory=list, description="What the agent can do next."
    )


class ERTriageState(State):
    """
    State for the ER Triage environment.
    Tracks episode metadata, including the current task and bias information.
    """
    task: str
    patient_queue: List[Dict]
    current_patient_index: int
    steps_taken_for_patient: int
    bias_log: List[Dict] = Field(
        default_factory=list,
        description="Tracks demographic vs. decision for bias detection."
    )
