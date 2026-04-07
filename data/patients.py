# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Synthetic patient data for the ER Triage environment.

Based on the public Emergency Severity Index (ESI) handbook and vignette libraries.
"""

PATIENTS = [
    # CRITICAL (ESI 1-2)
    {
        "patient_id": "P001",
        "chief_complaint": "Chest pain, shortness of breath",
        "vitals": {"hr": 120, "bp": "90/60", "rr": 28, "temp": 37.0, "spo2": 88},
        "history": "History of heart disease",
        "demographics": {"age": 65, "gender": "Male"},
        "ground_truth_esi": 1,
        "ground_truth_priority": "critical",
        "ideal_steps": 1,
    },
    {
        "patient_id": "P002",
        "chief_complaint": "Altered mental status",
        "vitals": {"hr": 110, "bp": "100/70", "rr": 22, "temp": 38.5, "spo2": 92},
        "history": "Diabetes",
        "demographics": {"age": 72, "gender": "Female"},
        "ground_truth_esi": 2,
        "ground_truth_priority": "critical",
        "ideal_steps": 2,
    },
    {
        "patient_id": "P003",
        "chief_complaint": "Severe abdominal pain",
        "vitals": {"hr": 115, "bp": "110/70", "rr": 20, "temp": 37.2, "spo2": 98},
        "history": "None",
        "demographics": {"age": 34, "gender": "Male"},
        "ground_truth_esi": 2,
        "ground_truth_priority": "critical",
        "ideal_steps": 2,
    },

    # URGENT (ESI 3)
    {
        "patient_id": "P004",
        "chief_complaint": "Headache and dizziness after falling",
        "vitals": {"hr": 80, "bp": "130/85", "rr": 18, "temp": 37.1, "spo2": 99},
        "history": "On blood thinners",
        "demographics": {"age": 78, "gender": "Female"},
        "ground_truth_esi": 3,
        "ground_truth_priority": "urgent",
        "ideal_steps": 3,
    },
    {
        "patient_id": "P005",
        "chief_complaint": "Cough and fever for 3 days",
        "vitals": {"hr": 95, "bp": "120/80", "rr": 20, "temp": 38.8, "spo2": 95},
        "history": "Asthma",
        "demographics": {"age": 45, "gender": "Male"},
        "ground_truth_esi": 3,
        "ground_truth_priority": "urgent",
        "ideal_steps": 2,
    },

    # NON-URGENT (ESI 4-5)
    {
        "patient_id": "P006",
        "chief_complaint": "Sore throat and runny nose",
        "vitals": {"hr": 70, "bp": "110/70", "rr": 16, "temp": 37.5, "spo2": 99},
        "history": "None",
        "demographics": {"age": 25, "gender": "Female"},
        "ground_truth_esi": 5,
        "ground_truth_priority": "non-urgent",
        "ideal_steps": 1,
    },
    {
        "patient_id": "P007",
        "chief_complaint": "Twisted ankle",
        "vitals": {"hr": 85, "bp": "120/80", "rr": 18, "temp": 37.0, "spo2": 100},
        "history": "None",
        "demographics": {"age": 30, "gender": "Male"},
        "ground_truth_esi": 4,
        "ground_truth_priority": "non-urgent",
        "ideal_steps": 2,
    },
    {
        "patient_id": "P008",
        "chief_complaint": "Needs a prescription refill",
        "vitals": {"hr": 75, "bp": "115/75", "rr": 16, "temp": 37.0, "spo2": 98},
        "history": "Hypertension (stable)",
        "demographics": {"age": 68, "gender": "Male"},
        "ground_truth_esi": 5,
        "ground_truth_priority": "non-urgent",
        "ideal_steps": 1,
    },

    # TRICKY / DIFFERENTIAL (for hard task)
    {
        "patient_id": "P009",
        "chief_complaint": "Mild headache, feeling a bit tired",
        "vitals": {"hr": 105, "bp": "85/55", "rr": 24, "temp": 38.9, "spo2": 91},
        "history": "Recent travel to tropical region, no prior medical issues",
        "demographics": {"age": 28, "gender": "Female"},
        "ground_truth_esi": 2,
        "ground_truth_priority": "critical",
        "ideal_steps": 2,
        "tricky": True,
    },
    {
        "patient_id": "P010",
        "chief_complaint": "Stomach ache after eating",
        "vitals": {"hr": 130, "bp": "80/50", "rr": 26, "temp": 37.1, "spo2": 93},
        "history": "Known aortic aneurysm",
        "demographics": {"age": 70, "gender": "Male"},
        "ground_truth_esi": 1,
        "ground_truth_priority": "critical",
        "ideal_steps": 2,
        "tricky": True,
    },
    {
        "patient_id": "P011",
        "chief_complaint": "Feeling anxious, heart racing",
        "vitals": {"hr": 88, "bp": "125/80", "rr": 18, "temp": 37.0, "spo2": 99},
        "history": "Generalized anxiety disorder, on SSRIs",
        "demographics": {"age": 32, "gender": "Female"},
        "ground_truth_esi": 4,
        "ground_truth_priority": "non-urgent",
        "ideal_steps": 2,
        "tricky": True,
    },
]
