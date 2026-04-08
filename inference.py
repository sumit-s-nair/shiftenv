#!/usr/bin/env python3
"""
inference.py — Tool-calling ShiftEnv agent.

This script emits mandatory structured logs:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Environment imports
from server.app import MigrationEnv, MigrationAction
from tools.lib_inspect import get_equivalent_func
from tools.func_diff import get_func_diff, format_diff_for_agent
from tools.sandbox import pip_install
from tools.call_graph import get_call_graph

# ---------------------------------------------------------
# Config & Environment Variables
# ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
HF_TOKEN = os.getenv("HF_TOKEN") or ""
TASK_NAME = os.getenv("OPENENV_TASK", "medium")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "shiftenv_v1")
MAX_STEPS = 10
KEEP_WORKDIR = False
SUCCESS_SCORE_THRESHOLD = 0.90
TOOL_LOOP_LIMIT = 8

# ---------------------------------------------------------
# Mandatory Logging Format (STDOUT)
# ---------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure action string doesn't contain newlines to keep log on one line
    action_clean = action.replace("\n", " ")[:100] 
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# Debug logging to STDERR (Validator ignores stderr)
def debug_log(msg: str):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

# ---------------------------------------------------------
# OpenAI Client & Tools
# ---------------------------------------------------------
def create_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or HF_TOKEN
    if not api_key:
        raise EnvironmentError("Missing API key: set HF_TOKEN or OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_call_graph",
            "description": "Get an AST-based dependency graph showing which functions use the target library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_lib": {"type": "string", "description": "The old library name"}
                },
                "required": ["target_lib"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_equivalent_func",
            "description": "Find the equivalent function name in the new library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_lib": {"type": "string"},
                    "old_func": {"type": "string"},
                    "new_lib": {"type": "string"},
                    "failed": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["old_lib", "old_func", "new_lib"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_func_diff",
            "description": "Get signature and behavioral differences between old and new functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_lib": {"type": "string"},
                    "new_lib": {"type": "string"},
                    "old_func": {"type": "string"},
                    "new_func": {"type": "string"}
                },
                "required": ["old_lib", "new_lib", "old_func"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Install a library at runtime.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lib": {"type": "string"}
                },
                "required": ["lib"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Write migrated code to a file and trigger test execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "new_code": {"type": "string", "description": "Full file content"}
                },
                "required": ["file_path", "new_code"]
            }
        }
    }
]

def dispatch_tool(tool_name: str, args: dict, repo_dir: str = ".") -> str:
    debug_log(f"Calling tool: {tool_name}")
    try:
        if tool_name == "get_call_graph":
            result = get_call_graph(repo_dir=repo_dir, target_lib=args["target_lib"])
            return json.dumps(result, default=str)
        elif tool_name == "get_equivalent_func":
            result = get_equivalent_func(old_lib=args["old_lib"], old_func=args["old_func"], 
                                       new_lib=args["new_lib"], failed=args.get("failed"))
            return json.dumps(result, default=str)
        elif tool_name == "get_func_diff":
            diff = get_func_diff(old_lib=args["old_lib"], new_lib=args["new_lib"], 
                               func_path=args["old_func"], new_func_path=args.get("new_func"))
            return format_diff_for_agent(diff)
        elif tool_name == "pip_install":
            result = pip_install(lib=args["lib"])
            return json.dumps(result, default=str)
        elif tool_name == "edit_file":
            return "EDIT_PENDING"
    except Exception as e:
        return json.dumps({"error": str(e)})
    return "Unknown tool"


def _sanitize_generated_code(new_code: str, fallback: str) -> str:
    code = (new_code or "").strip()
    if "```" in code:
        # Keep behavior tolerant to markdown-fenced outputs.
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].replace("python", "", 1).strip()
    return code or fallback


def _compute_episode_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    raw_score = sum(rewards) / len(rewards)
    return max(0.0, min(1.0, round(raw_score, 4)))

# ---------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are a Python migration agent.
STRICT PROCESS:
1. Call get_call_graph to map dependencies.
2. Call get_equivalent_func and get_func_diff for every function you migrate.
3. Call edit_file with the full raw Python code.
Output ONLY raw code in the edit_file tool call."""

def identify_files_to_migrate(context: str, old_lib: str) -> list[str]:
    files = []
    current_file = None
    for line in context.splitlines():
        if line.startswith("--- ") and line.endswith(" ---"):
            current_file = line.strip("- ").replace("(current)", "").replace("(MIGRATE THIS FILE)", "").replace("(reference only, do NOT output this)", "").strip()
        elif current_file and (f"import {old_lib}" in line or f"from {old_lib}" in line):
            if current_file not in files: files.append(current_file)
    return files


def run_task(task_name: str) -> None:
    env = MigrationEnv(max_steps=MAX_STEPS)

    # Global tracking for [END] log
    rewards_list = []
    global_step = 0
    success = False
    final_score = 0.0

    log_start(task_name, BENCHMARK, MODEL_NAME)

    try:
        obs = env.reset(task_name=task_name)
        if obs.status == "error":
            raise Exception(f"Env reset failed: {obs.message}")

        client = create_client()
        state = env.state
        files_to_migrate = identify_files_to_migrate(obs.context, state.old_lib)
        if not files_to_migrate:
            files_to_migrate = [f for f in state.files_in_repo if not f.startswith("__")]

        for file_path in files_to_migrate:
            if env.done or global_step >= MAX_STEPS: break

            target_file_path = os.path.join(env._work_repo, file_path)
            try:
                with open(target_file_path, "r") as f:
                    content = f.read()
            except Exception as e:
                debug_log(f"Could not read {file_path}: {e}")
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Migrate `{file_path}` from `{state.old_lib}` to `{state.new_lib}`.\n\nCode:\n{content}"}
            ]

            edited_this_file = False
            # Keep iterating until the model performs an edit or we hit guard rails.
            for _ in range(TOOL_LOOP_LIMIT):
                if env.done: break

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=0.1,
                    )
                    assistant_msg = response.choices[0].message
                except Exception as e:
                    debug_log(f"Model call failed on {file_path}: {e}")
                    messages.append({"role": "user", "content": "Proceed with an edit_file tool call now."})
                    continue

                messages.append(assistant_msg.model_dump(exclude_none=True))

                if assistant_msg.tool_calls:
                    saw_non_edit_tool = False
                    for tool_call in assistant_msg.tool_calls:
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except Exception:
                            fn_args = {}
                        
                        if fn_name == "edit_file":
                            global_step += 1
                            new_code = _sanitize_generated_code(fn_args.get("new_code", ""), fallback=content)

                            action = MigrationAction(file_path=file_path, new_code=new_code)
                            try:
                                obs, reward, done, info = env.step(action)
                            except Exception as e:
                                log_step(global_step, f"edit:{file_path}", 0.0, True, str(e))
                                raise

                            rewards_list.append(reward)
                            final_score = _compute_episode_score(rewards_list)
                            edited_this_file = True
                            
                            log_step(global_step, f"edit:{file_path}", reward, done, None)
                            
                            if reward < 0.99:
                                messages.append({"role": "user", "content": f"Tests failed. Output: {info.get('test_result', {}).get('output', '')[:500]}"})
                            break
                        else:
                            saw_non_edit_tool = True
                            res = dispatch_tool(fn_name, fn_args, repo_dir=env._work_repo)
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res})

                    if edited_this_file:
                        break
                    if saw_non_edit_tool:
                        continue
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must now call edit_file with the full migrated Python file content. "
                            "Do not ask questions."
                        ),
                    })

            if not edited_this_file and not env.done and global_step < MAX_STEPS:
                # Last-resort: force an edit_file tool call so the episode can progress.
                try:
                    forced = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice={"type": "function", "function": {"name": "edit_file"}},
                        temperature=0.0,
                    )
                    forced_msg = forced.choices[0].message
                    messages.append(forced_msg.model_dump(exclude_none=True))
                except Exception as e:
                    debug_log(f"Forced edit_file call failed on {file_path}: {e}")
                    forced_msg = None

                if forced_msg and forced_msg.tool_calls:
                    for tool_call in forced_msg.tool_calls:
                        if tool_call.function.name != "edit_file":
                            continue
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except Exception:
                            fn_args = {}
                        new_code = _sanitize_generated_code(fn_args.get("new_code", ""), fallback=content)

                        global_step += 1
                        action = MigrationAction(file_path=file_path, new_code=new_code)
                        obs, reward, done, info = env.step(action)
                        rewards_list.append(reward)
                        final_score = _compute_episode_score(rewards_list)
                        log_step(global_step, f"edit:{file_path}", reward, done, None)
                        edited_this_file = True
                        if reward < 0.99:
                            messages.append({"role": "user", "content": f"Tests failed. Output: {info.get('test_result', {}).get('output', '')[:500]}"})
                        break

            if not edited_this_file:
                debug_log(f"No edit_file call produced for {file_path}; skipping file.")

        final_score = _compute_episode_score(rewards_list)
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        debug_log(f"Critical Failure: {e}")
    finally:
        final_score = _compute_episode_score(rewards_list)
        # Mandatory: Always emit [END]
        log_end(success, global_step, final_score, rewards_list)
        if not KEEP_WORKDIR:
            try:
                env.cleanup()
            except Exception as e:
                debug_log(f"Cleanup error: {e}")


def run_agent() -> None:
    run_task(TASK_NAME)

if __name__ == "__main__":
    run_agent()