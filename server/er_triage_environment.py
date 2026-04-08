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
from typing import Dict, List, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ERTriageAction, ERTriageObservation, ERTriageState
    from ..data.patients import PATIENTS
except ImportError:
    from models import ERTriageAction, ERTriageObservation, ERTriageState
    from data.patients import PATIENTS


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

    def reset(self, task: str = None, **kwargs) -> ERTriageObservation:
        """Reset the environment to start a new episode."""
        if task:
            self._task = task
        self._state = self._create_initial_state()

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

        return self._get_observation()

    def _get_observation(self) -> ERTriageObservation:
        """Constructs the observation for the current patient."""
        if self._state.current_patient_index >= len(self._state.patient_queue):
            return ERTriageObservation(
                patient_id="",
                chief_complaint="No more patients in the queue.",
                done=True,
                reward=0.0,
                available_actions=[]
            )

        patient = self._state.patient_queue[self._state.current_patient_index]

        return ERTriageObservation(
            patient_id=patient["patient_id"],
            chief_complaint=patient["chief_complaint"],
            available_actions=["request_vitals", "ask_question", "assign_priority"],
            done=False,
            reward=0.0,
        )

    def step(self, action: ERTriageAction) -> ERTriageObservation:
        """Execute a step in the environment."""
        self._state.step_count += 1
        self._state.steps_taken_for_patient += 1

        patient = self._state.patient_queue[self._state.current_patient_index]
        obs = self._get_observation()
        reward = 0.0
        done = False

        if action.action_type == "request_vitals":
            obs.vitals = patient["vitals"]
            obs.available_actions = ["ask_question", "assign_priority"]
            reward = 0.2

        elif action.action_type == "ask_question":
            obs.question_answer = f"Patient responds to '{action.question}'. History: {patient.get('history', 'N/A')}"
            obs.available_actions = ["assign_priority"]
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
                self._state.current_patient_index += 1
                self._state.steps_taken_for_patient = 0
                obs = self._get_observation()

        if self._state.steps_taken_for_patient >= 3:
            obs.available_actions = ["assign_priority"]

        obs.reward = reward
        obs.done = done
        return obs

    def _grade_priority(self, assigned_priority: str, patient: Dict) -> Tuple[float, bool]:
        """Grades the assigned priority. Returns reward in [0, 0.7]."""
        ground_truth = patient["ground_truth_priority"]
        correct = assigned_priority == ground_truth

        if correct:
            reward = 0.7
        else:
            reward = 0.0

        return reward, correct

    @property
    def state(self) -> ERTriageState:
        """Get the current environment state."""
        return self._state
