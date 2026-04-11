# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ER Triage Environment Implementation.

An environment where an agent triages emergency room patients using the
Emergency Severity Index (ESI) protocol.
"""

import random
from uuid import uuid4
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ERTriageAction, ERTriageObservation, ERTriageState
    from ..data.patients import PATIENTS
except ImportError:
    from models import ERTriageAction, ERTriageObservation, ERTriageState
    from data.patients import PATIENTS


# Priority distance matrix for partial credit grading.
# Adjacent levels get partial credit; two levels away gets less.
_PRIORITY_LEVELS = ["non-urgent", "urgent", "critical"]
_PRIORITY_INDEX = {p: i for i, p in enumerate(_PRIORITY_LEVELS)}


class ERTriageEnvironment(Environment[ERTriageAction, ERTriageObservation, ERTriageState]):
    """
    ER Triage Environment.

    The agent's goal is to correctly assign an ESI priority level to patients
    in a sequential manner, balancing speed and accuracy.

    Tasks:
        - single_triage: Triage 1 patient (easy)
        - batch_triage: Triage 3 patients sequentially (medium)
        - differential_triage: Triage 1 tricky patient with misleading symptoms (hard)
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "single_triage"):
        """Initialize the ER Triage environment."""
        self._task = task
        self._state: ERTriageState = self._create_initial_state()
        # Accumulated info for the current patient encounter
        self._patient_vitals: Optional[Dict] = None
        self._patient_question_answers: List[str] = []
        self._patient_history: Optional[str] = None

    def _create_initial_state(self) -> ERTriageState:
        """Creates the initial state for a new episode."""
        return ERTriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task=self._task,
            patient_queue=[],
            current_patient_index=0,
            steps_taken_for_patient=0,
            bias_log=[],
        )

    def _reset_patient_context(self) -> None:
        """Clear accumulated context when moving to a new patient."""
        self._patient_vitals = None
        self._patient_question_answers = []
        self._patient_history = None

    def reset(self, task: str = None, **kwargs) -> ERTriageObservation:
        """Reset the environment to start a new episode."""
        if task:
            self._task = task
        self._state = self._create_initial_state()
        self._reset_patient_context()

        if self._task == "batch_triage":
            self._state.patient_queue = random.sample(PATIENTS, min(3, len(PATIENTS)))
        elif self._task == "differential_triage":
            tricky = [p for p in PATIENTS if p.get("tricky", False)]
            if tricky:
                self._state.patient_queue = [random.choice(tricky)]
            else:
                self._state.patient_queue = [random.choice(PATIENTS)]
        else:
            # single_triage (default)
            self._state.patient_queue = [random.choice(PATIENTS)]

        self._state.current_patient_index = 0
        self._state.steps_taken_for_patient = 0

        return self._build_observation(reward=0.0, done=False)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp reward/score to open interval (0, 1) as required by validator."""
        eps = 0.001
        return min(max(value, eps), 1.0 - eps)

    def _build_observation(self, reward: float, done: bool) -> ERTriageObservation:
        """Build observation with full accumulated patient context."""
        if self._state.current_patient_index >= len(self._state.patient_queue):
            return ERTriageObservation(
                patient_id="",
                chief_complaint="No more patients in the queue.",
                done=True,
                reward=self._clamp(reward),
                available_actions=[],
            )

        patient = self._state.patient_queue[self._state.current_patient_index]

        # Build available actions based on what info has been gathered
        actions = []
        if self._patient_vitals is None:
            actions.append("request_vitals")
        if self._state.steps_taken_for_patient < 3:
            actions.append("ask_question")
        actions.append("assign_priority")

        # Force decision after 3 info-gathering steps
        if self._state.steps_taken_for_patient >= 3:
            actions = ["assign_priority"]

        # Combine all question answers into a single context string
        qa_context = None
        if self._patient_question_answers:
            qa_context = " | ".join(self._patient_question_answers)

        return ERTriageObservation(
            patient_id=patient["patient_id"],
            chief_complaint=patient["chief_complaint"],
            vitals=self._patient_vitals,
            history=self._patient_history,
            question_answer=qa_context,
            available_actions=actions,
            done=done,
            reward=self._clamp(reward),
        )

    def step(self, action: ERTriageAction) -> ERTriageObservation:
        """Execute a step in the environment."""
        self._state.step_count += 1
        self._state.steps_taken_for_patient += 1

        patient = self._state.patient_queue[self._state.current_patient_index]
        reward = 0.0
        done = False

        if action.action_type == "request_vitals":
            if self._patient_vitals is None:
                self._patient_vitals = patient["vitals"]
                reward = 0.2
            else:
                # Already requested — no additional reward
                reward = 0.0

        elif action.action_type == "ask_question":
            question = action.question or "Tell me more about your symptoms."
            answer = f"Q: '{question}' — {patient.get('history', 'No relevant history.')}"
            self._patient_question_answers.append(answer)
            self._patient_history = patient.get("history", "N/A")
            reward = 0.1

        elif action.action_type == "assign_priority":
            reward, correct = self._grade_priority(action.priority, patient)

            self._state.bias_log.append({
                "demographics": patient["demographics"],
                "assigned_priority": action.priority,
                "ground_truth": patient["ground_truth_priority"],
                "correct": correct,
            })

            if self._state.current_patient_index >= len(self._state.patient_queue) - 1:
                done = True
            else:
                # Move to next patient
                self._state.current_patient_index += 1
                self._state.steps_taken_for_patient = 0
                self._reset_patient_context()

        return self._build_observation(reward=reward, done=done)

    def _grade_priority(self, assigned_priority: str, patient: Dict) -> Tuple[float, bool]:
        """Grade the assigned priority with partial credit for near-misses.

        Reward structure:
            Exact match:           0.70
            One level off:         0.30  (e.g., urgent vs critical)
            Two levels off:        0.05  (e.g., non-urgent vs critical)
        """
        ground_truth = patient["ground_truth_priority"]
        correct = assigned_priority == ground_truth

        if correct:
            return 0.70, True

        gt_idx = _PRIORITY_INDEX.get(ground_truth, 1)
        assigned_idx = _PRIORITY_INDEX.get(assigned_priority, 1)
        distance = abs(gt_idx - assigned_idx)

        if distance == 1:
            return 0.30, False  # adjacent level — partial credit
        return 0.05, False      # two levels off — minimal credit

    @property
    def state(self) -> ERTriageState:
        """Get the current environment state."""
        return self._state
