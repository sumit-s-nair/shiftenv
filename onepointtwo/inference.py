#!/usr/bin/env python3
"""
inference.py — Code Migration OpenEnv Agent
===========================================
Runs an LLM agent across migration tasks and emits structured stdout logs.

Stdout format (must not deviate):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
from typing import List, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

ENV_BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "CodeMigration-v1"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.5

TASKS = [
    "task_easy",
    "task_medium",
    "task_hard",
]

# ---------------------------------------------------------------------------
# Logging (STRICT FORMAT)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()

    action_clean = action.replace("\n", "\\n")[:100]

    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment Client
# ---------------------------------------------------------------------------

class CodeMigrationClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)
        self.session_id: Optional[str] = None

    def reset(self, task_id: str) -> dict:
        r = self.client.post(f"{self.base_url}/reset", params={"task_id": task_id})
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        return data["observation"]

    def step(self, action: Dict) -> dict:
        r = self.client.post(
            f"{self.base_url}/step",
            params={"session_id": self.session_id},
            json=action,
        )
        r.raise_for_status()
        return r.json()

    def grade(self) -> dict:
        r = self.client.get(
            f"{self.base_url}/grader",
            params={"session_id": self.session_id},
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()

# ---------------------------------------------------------------------------
# Agent Logic (WITH TAVILY)
# ---------------------------------------------------------------------------

class MigrationAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.cache: Dict[str, str] = {}

    def find_equivalent(self, old_call: str, target_lib: str) -> str:
        key = f"{old_call}->{target_lib}"
        if key in self.cache:
            return self.cache[key]

        query = f"Python: what is the {target_lib} equivalent of {old_call}?"

        try:
            resp = self.client.responses.create(
                model=MODEL_NAME,
                tools=[{
                    "type": "mcp",
                    "server_label": "tavily",
                    "server_url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
                    "require_approval": "never"
                }],
                input=[
                    {"role": "system", "content": "Return JSON: {'equivalent_function': 'name'}"},
                    {"role": "user", "content": query},
                ],
            )

            text = resp.output_text.strip()
            text = text.replace("```json", "").replace("```", "")

            data = json.loads(text)
            result = data.get("equivalent_function", old_call)

            self.cache[key] = result
            return result

        except Exception:
            # Fallback: direct LLM (no tool)
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": f"What is the {target_lib} equivalent of {old_call}? Return only function name."
                    }],
                    temperature=0.0,
                )
                result = resp.choices[0].message.content.strip()
                self.cache[key] = result
                return result
            except:
                return old_call

    def rewrite(self, code: str, mapping: Dict[str, str], target_lib: str) -> str:
        mapping_str = "\n".join([f"{k} -> {v}" for k, v in mapping.items()])

        prompt = (
            f"Rewrite this Python code to use {target_lib}.\n\n"
            f"Mappings:\n{mapping_str}\n\n"
            f"Return ONLY valid Python code.\n\n"
            f"{code}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return (
                resp.choices[0].message.content
                .replace("```python", "")
                .replace("```", "")
                .strip()
            )
        except:
            return code

# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, env: CodeMigrationClient, task_name: str) -> None:
    agent = MigrationAgent(client)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_name)

        for step in range(1, MAX_STEPS + 1):
            pending = obs.get("pending_functions", [])
            if not pending:
                break

            func = pending[0]
            target_lib = obs.get("target_library", "")

            old_calls = func.get("library_functions_used", [])

            mapping = {
                call: agent.find_equivalent(call, target_lib)
                for call in old_calls
            }

            new_code = agent.rewrite(func["code"], mapping, target_lib)

            try:
                result = env.step({
                    "func_id": func["func_id"],
                    "rewritten_code": new_code,
                })
                error = None
            except Exception as exc:
                error = str(exc)
                break

            obs = result["observation"]
            reward = result["reward"]["value"]
            done = result["done"]

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(mapping),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        grade = env.grade()
        score = grade.get("score", 0.0)
        success = grade.get("success", False)

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} failed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[ERROR] Missing API key.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CodeMigrationClient(ENV_BASE_URL)

    try:
        for task in TASKS:
            run_task(client, env, task)
    finally:
        env.close()


if __name__ == "__main__":
    main()