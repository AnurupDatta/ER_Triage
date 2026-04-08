"""
Inference Script for ER Triage Environment
===========================================
MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  Docker image name (optional, if using from_docker_image).

STDOUT FORMAT:
    [START] task=<task_name> env=er_triage model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Tasks (easy -> medium -> hard):
    single_triage:       Triage 1 patient (easy)
    batch_triage:        Triage 3 patients sequentially (medium)
    differential_triage: Triage 1 tricky patient with misleading symptoms (hard)
"""

import asyncio
import json
import os
import textwrap
import time
from typing import Dict, List, Optional

from openai import OpenAI

try:
    from client import ERTriageEnv
    from models import ERTriageAction, ERTriageObservation
except ImportError:
    from .client import ERTriageEnv
    from .models import ERTriageAction, ERTriageObservation

# ── Environment Variables ──────────────────────────────────────────────────────
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "https://hawkaii-er-triage.hf.space")
TASK_NAME = os.getenv("TASK_NAME")  # None = run all tasks
BENCHMARK = "er_triage"

# ── Task Configuration ─────────────────────────────────────────────────────────
TASKS = ["single_triage", "batch_triage", "differential_triage"]

TASK_CONFIG: Dict[str, Dict] = {
    "single_triage":       {"num_patients": 1, "max_steps": 4},
    "batch_triage":        {"num_patients": 3, "max_steps": 12},
    "differential_triage": {"num_patients": 1, "max_steps": 4},
}

# Max reward per patient: request_vitals(0.2) + ask_question(0.1) + correct_priority(0.7) = 1.0
MAX_REWARD_PER_PATIENT = 1.0
SUCCESS_SCORE_THRESHOLD = 0.1

TEMPERATURE = 0.3
MAX_TOKENS = 300

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an emergency room triage nurse using the Emergency Severity Index (ESI).

    For each patient, decide your next action. Respond with ONLY a JSON object.

    Actions:
    1. Request vitals: {"action_type": "request_vitals", "reasoning": "why"}
    2. Ask a question: {"action_type": "ask_question", "question": "your question", "reasoning": "why"}
    3. Assign priority: {"action_type": "assign_priority", "priority": "critical|urgent|non-urgent", "reasoning": "why"}

    Priority levels:
    - "critical": ESI 1-2. Life-threatening. Examples: chest pain + unstable vitals, altered mental status, SpO2<94, HR>110 with hypotension.
    - "urgent": ESI 3. Needs multiple resources. Examples: fall + blood thinners, persistent fever + respiratory history, needs CT/labs.
    - "non-urgent": ESI 4-5. Minor issue. Examples: sore throat, twisted ankle, prescription refill, stable anxiety.

    Strategy:
    - Always request vitals first if not yet available.
    - Use vitals to guide urgency: HR>100, SpO2<94, RR>24, BP systolic<90 suggest critical.
    - If vitals are borderline, ask a clarifying question about history.
    - Do NOT be fooled by mild-sounding complaints — always check vitals before deciding.
    - Assign priority as soon as you have enough information.
""").strip()


# ── Structured Logging ─────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM Interaction ───────────────────────────────────────────────────────────
def build_user_prompt(obs: ERTriageObservation, step: int, history: List[str]) -> str:
    parts = [f"Step: {step}"]
    parts.append(f"Patient ID: {obs.patient_id}")
    parts.append(f"Chief Complaint: {obs.chief_complaint}")

    if obs.vitals:
        parts.append(f"Vitals: HR={obs.vitals.get('hr')}, BP={obs.vitals.get('bp')}, "
                      f"RR={obs.vitals.get('rr')}, Temp={obs.vitals.get('temp')}, "
                      f"SpO2={obs.vitals.get('spo2')}")
    else:
        parts.append("Vitals: Not yet requested")

    if obs.question_answer:
        parts.append(f"Previous Answer: {obs.question_answer}")

    parts.append(f"Available Actions: {obs.available_actions}")

    if history:
        parts.append("Previous steps:\n" + "\n".join(history[-4:]))

    parts.append("\nRespond with a JSON object for your next action.")
    return "\n".join(parts)


def parse_llm_response(text: str) -> ERTriageAction:
    """Parse LLM JSON response into ERTriageAction."""
    try:
        clean = text.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        data = json.loads(clean)
        return ERTriageAction(
            action_type=data.get("action_type", "request_vitals"),
            question=data.get("question"),
            priority=data.get("priority"),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        text_lower = text.lower()
        if "assign_priority" in text_lower:
            priority = "urgent"
            for p in ["critical", "non-urgent", "urgent"]:
                if p in text_lower:
                    priority = p
                    break
            return ERTriageAction(action_type="assign_priority", priority=priority, reasoning="parse fallback")
        elif "ask_question" in text_lower:
            return ERTriageAction(action_type="ask_question", question="Can you describe your symptoms in more detail?", reasoning="parse fallback")
        else:
            return ERTriageAction(action_type="request_vitals", reasoning="parse fallback")


def get_llm_action(client: OpenAI, obs: ERTriageObservation, step: int, history: List[str]) -> ERTriageAction:
    """Get next action from the LLM."""
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ERTriageAction(action_type="request_vitals", reasoning=f"LLM error fallback: {exc}")


# ── Run a single task ─────────────────────────────────────────────────────────
async def run_task(llm_client: OpenAI, env: ERTriageEnv, task_name: str) -> float:
    """Run one full episode for a task. Returns the final score."""
    config = TASK_CONFIG.get(task_name, TASK_CONFIG["single_triage"])
    num_patients = config["num_patients"]
    max_steps = config["max_steps"]
    max_total_reward = num_patients * MAX_REWARD_PER_PATIENT

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for patient_idx in range(num_patients):
            result = await env.reset(params={"task": task_name})
            obs = result.observation
            done = result.done
            patient_history: List[str] = []

            while not done and steps_taken < max_steps:
                action = get_llm_action(llm_client, obs, steps_taken + 1, patient_history)

                result = await env.step(action)
                obs = result.observation
                done = result.done
                reward = result.reward or 0.0
                error = None

                steps_taken += 1
                rewards.append(reward)

                action_str = action.action_type
                if action.action_type == "assign_priority":
                    action_str = f"assign_priority({action.priority})"
                elif action.action_type == "ask_question":
                    q = (action.question or "")[:50]
                    action_str = f"ask_question('{q}')"

                is_final_done = done and (patient_idx == num_patients - 1)

                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward,
                    done=is_final_done,
                    error=error,
                )

                patient_history.append(f"Step {steps_taken}: {action_str} -> reward {reward:+.2f}")
                history.append(patient_history[-1])

                if done:
                    break

        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        # Clamp to open interval (0, 1) — validator requires strictly between 0 and 1
        eps = 1e-6
        score = min(max(score, eps), 1.0 - eps)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ───────────────────────────────────────────────────────────────────────
async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await ERTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = ERTriageEnv(base_url=ENV_URL)

    tasks_to_run = [TASK_NAME] if TASK_NAME else TASKS
    scores: Dict[str, float] = {}

    try:
        for task_name in tasks_to_run:
            try:
                score = await run_task(llm_client, env, task_name)
                scores[task_name] = score
            except Exception as exc:
                print(f"[ERROR] task={task_name} error={exc}", flush=True)
                scores[task_name] = 0.0
            time.sleep(1)

        print(json.dumps({
            "event": "[SUMMARY]",
            "scores": scores,
            "average": round(sum(scores.values()) / len(scores), 4),
        }), flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
