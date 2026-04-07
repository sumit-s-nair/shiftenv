#!/usr/bin/env python3
"""
inference.py — Tool-calling agent for ShiftEnv (OpenEnv hackathon).

Uses OpenAI function-calling to invoke tools before making edits.
Tools: get_equivalent_func, get_func_diff, pip_install, edit_file.

Outputs structured JSON logs to stdout:
    {"type": "[START]", "task": "..."}
    {"type": "[STEP]",  "step": N, "action": "...", "reward": R}
    {"type": "[END]",   "task": "...", "score": S}
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from env import MigrationEnv, MigrationAction
from tools.lib_inspect import get_equivalent_func
from tools.func_diff import get_func_diff, format_diff_for_agent
from tools.sandbox import pip_install
from tools.call_graph import get_call_graph

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENENV_TASK = os.environ.get("OPENENV_TASK", "medium")
MAX_STEPS = 10
KEEP_WORKDIR = False


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
def log(data: dict):
    print(json.dumps(data), flush=True)

def log_start(task: str):
    log({"type": "[START]", "task": task})

def log_step(step: int, action: str, reward: float):
    log({"type": "[STEP]", "step": step, "action": action, "reward": reward})

def log_end(task: str, score: float):
    log({"type": "[END]", "task": task, "score": score})

def log_tool(name: str, args: dict):
    print(f"[TOOL] {name}({json.dumps(args, default=str)[:300]})", file=sys.stderr)

def log_tool_result(result: str):
    print(f"[TOOL_RESULT] {result[:200]}", file=sys.stderr)


# ---------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------
def create_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or HF_TOKEN
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


# ---------------------------------------------------------
# Tool Definitions (OpenAI function-calling format)
# ---------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_call_graph",
            "description": (
                "Get an AST-based dependency graph showing which functions and classes "
                "in this codebase use the given target library. Use this to understand "
                "exactly what needs to be migrated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_lib": {
                        "type": "string",
                        "description": "The old library name (e.g. 'tensorflow')"
                    }
                },
                "required": ["target_lib"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_equivalent_func",
            "description": (
                "Find the equivalent function name in the new library. "
                "Call this FIRST for each function/class you need to migrate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_lib": {
                        "type": "string",
                        "description": "The old library name (e.g. 'tensorflow', 'requests', 'random')"
                    },
                    "old_func": {
                        "type": "string",
                        "description": "The function/class path in the old library (e.g. 'keras.layers.Dense', 'Session', 'choice')"
                    },
                    "new_lib": {
                        "type": "string",
                        "description": "The new library name (e.g. 'torch', 'httpx', 'secrets')"
                    },
                    "failed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Previously tried wrong names — so the tool doesn't repeat them"
                    }
                },
                "required": ["old_lib", "old_func", "new_lib"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_func_diff",
            "description": (
                "Get full signature and behavioral differences between old and new "
                "library function. Always call this after get_equivalent_func to verify "
                "the mapping before writing code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_lib": {
                        "type": "string",
                        "description": "The old library name"
                    },
                    "new_lib": {
                        "type": "string",
                        "description": "The new library name"
                    },
                    "old_func": {
                        "type": "string",
                        "description": "Function path in the old library"
                    },
                    "new_func": {
                        "type": "string",
                        "description": "Function path in the new library (from get_equivalent_func result)"
                    }
                },
                "required": ["old_lib", "new_lib", "old_func"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": (
                "Install a library at runtime if not already present. "
                "Call this if get_func_diff returns 'not installed'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lib": {
                        "type": "string",
                        "description": "PyPI package name to install"
                    }
                },
                "required": ["lib"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Write migrated code to a file. ONLY call this AFTER using "
                "get_equivalent_func and get_func_diff to verify the migration. "
                "This triggers test execution and returns the reward."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file in the repo (e.g. 'model.py')"
                    },
                    "new_code": {
                        "type": "string",
                        "description": "Complete new file contents (the full migrated Python file)"
                    }
                },
                "required": ["file_path", "new_code"]
            }
        }
    }
]


# ---------------------------------------------------------
# Tool Dispatch
# ---------------------------------------------------------
def dispatch_tool(tool_name: str, args: dict, repo_dir: str = ".") -> str:
    """Call the actual tool and return a string result."""

    log_tool(tool_name, args)

    try:
        if tool_name == "get_call_graph":
            result = get_call_graph(repo_dir=repo_dir, target_lib=args["target_lib"])
            result_str = json.dumps(result, default=str)

        elif tool_name == "get_equivalent_func":
            result = get_equivalent_func(
                old_lib=args["old_lib"],
                old_func=args["old_func"],
                new_lib=args["new_lib"],
                failed=args.get("failed"),
            )
            result_str = json.dumps(result, default=str)

        elif tool_name == "get_func_diff":
            diff = get_func_diff(
                old_lib=args["old_lib"],
                new_lib=args["new_lib"],
                func_path=args["old_func"],
                new_func_path=args.get("new_func"),
            )
            result_str = format_diff_for_agent(diff)

        elif tool_name == "pip_install":
            result = pip_install(lib=args["lib"])
            result_str = json.dumps(result, default=str)

        elif tool_name == "edit_file":
            # edit_file is handled specially — return a marker
            result_str = "EDIT_PENDING"

        else:
            result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        result_str = json.dumps({"error": str(e)})

    log_tool_result(result_str)
    return result_str


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are a Python migration agent. Your job is to migrate Python codebases from one library to another.

STRICT PROCESS — follow this every time before editing any file:
1. Call get_call_graph to find out exactly where the old library is used in the repository.
2. Call get_equivalent_func to find the correct function name in the new library
3. Call get_func_diff to understand signature and behavioral differences
4. Only then call edit_file with the correctly migrated code

Do NOT rely on your memory of library APIs — the APIs may have changed or have subtle behavioral differences that will break tests. Always verify with tools first.

When tests fail after an edit:
- Read the error message carefully
- If the function signature is wrong → call get_func_diff again
- If you used the wrong function name → call get_equivalent_func with failed=[the wrong name you used]
- If the library isn't installed → call pip_install first

You must call get_equivalent_func and get_func_diff at least once per function you migrate. Do not skip this step.

Output ONLY raw Python code in edit_file new_code — no markdown, no backticks, no explanations."""


# ---------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------
def identify_files_to_migrate(context: str, old_lib: str) -> list[str]:
    """Parse the observation context to find files that import the old library."""
    files = []
    current_file = None
    for line in context.splitlines():
        if line.startswith("--- ") and line.endswith(" ---"):
            fname = line.strip("- ").strip()
            if fname.endswith("(current)"):
                fname = fname.replace("(current)", "").strip()
            if "(MIGRATE THIS FILE)" in fname:
                fname = fname.replace("(MIGRATE THIS FILE)", "").strip()
            if "(reference only, do NOT output this)" in fname:
                fname = fname.replace("(reference only, do NOT output this)", "").strip()
            current_file = fname
        elif current_file and (f"import {old_lib}" in line or f"from {old_lib}" in line):
            if current_file not in files:
                files.append(current_file)
    return files


def run_agent():
    """Main agent loop with tool-calling."""
    task_name = OPENENV_TASK
    log_start(task_name)

    # Initialize environment
    env = MigrationEnv(max_steps=MAX_STEPS)
    obs = env.reset(task_name=task_name)

    if obs.status == "error":
        print(f"[ERROR] Reset failed: {obs.message}", file=sys.stderr)
        log_end(task_name, 0.0)
        return

    client = create_client()
    state = env.state
    final_score = obs.reward
    initial_context = obs.context

    # Find files to migrate
    files_to_migrate = identify_files_to_migrate(initial_context, state.old_lib)
    if not files_to_migrate:
        files_to_migrate = [f for f in state.files_in_repo if not f.startswith("__")]

    print(f"[INFO] Task: {task_name} | {state.old_lib} → {state.new_lib}", file=sys.stderr)
    print(f"[INFO] Files to migrate: {files_to_migrate}", file=sys.stderr)

    step_num = 0

    for file_path in files_to_migrate:
        if env.done:
            break

        # Read the actual file content from the workdir
        target_file_path = os.path.join(env._work_repo, file_path)
        try:
            with open(target_file_path, "r") as f:
                current_file_content = f.read()
        except FileNotFoundError:
            print(f"[WARN] File not found: {target_file_path}", file=sys.stderr)
            continue

        # Build file-specific context
        file_context = f"--- {file_path} (MIGRATE THIS FILE) ---\n{current_file_content}\n"
        for other_file in state.files_in_repo:
            if other_file != file_path and not other_file.startswith("__"):
                other_path = os.path.join(env._work_repo, other_file)
                try:
                    with open(other_path, "r") as f:
                        other_content = f.read()
                    file_context += f"\n--- {other_file} (reference only, do NOT output this) ---\n{other_content}\n"
                except FileNotFoundError:
                    pass

        # Build initial user message for this file
        user_msg = (
            f"Migrate `{file_path}` from `{state.old_lib}` to `{state.new_lib}`.\n\n"
            f"Use tools to look up the correct API mappings before editing.\n\n"
            f"=== SOURCE CODE ===\n{file_context}"
        )

        # Initialize messages for this file's conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Try migrating this file (with retries if tests fail)
        attempts_for_file = 0
        max_attempts_per_file = min(3, MAX_STEPS - step_num)

        while attempts_for_file < max_attempts_per_file and not env.done:
            edit_action = None

            # ---- PHASE 1: Tool-calling loop ----
            tool_loop_iterations = 0
            max_tool_loops = 10  # safety cap
            nudge_sent = False   # only send the 'please edit now' nudge once

            while tool_loop_iterations < max_tool_loops:
                tool_loop_iterations += 1

                try:
                    start_time = time.time()
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        max_tokens=8192,
                        temperature=0.1,
                    )
                    elapsed = time.time() - start_time
                    print(f"[INFO] LLM call took {elapsed:.1f}s (tool loop iter {tool_loop_iterations})",
                          file=sys.stderr)
                except Exception as e:
                    print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                # Add assistant message to conversation
                messages.append(assistant_msg.model_dump(exclude_none=True))

                # No tool calls → agent is done reasoning (or gave up)
                if not assistant_msg.tool_calls:
                    # Check if the assistant responded with code directly
                    if assistant_msg.content and len(assistant_msg.content.strip()) > 50:
                        # Agent wrote code directly instead of using edit_file
                        content = assistant_msg.content.strip()
                        # Strip markdown fences
                        if content.startswith("```python"):
                            content = content[len("```python"):].strip()
                        if content.startswith("```"):
                            content = content[3:].strip()
                        if content.endswith("```"):
                            content = content[:-3].strip()
                        if len(content) > 50:
                            edit_action = MigrationAction(
                                file_path=file_path, new_code=content
                            )
                    break

                # Process each tool call
                for tool_call in assistant_msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    result_str = dispatch_tool(fn_name, fn_args, repo_dir=env._work_repo)

                    if fn_name == "edit_file":
                        # Build the action — strip markdown fences from code
                        new_code = fn_args.get("new_code", "")
                        new_code = new_code.strip()
                        if new_code.startswith("```python"):
                            new_code = new_code[len("```python"):].strip()
                        if new_code.startswith("```"):
                            new_code = new_code[3:].strip()
                        if new_code.endswith("```"):
                            new_code = new_code[:-3].strip()

                        fp = fn_args.get("file_path", file_path)
                        edit_action = MigrationAction(file_path=fp, new_code=new_code)
                        result_str = "Edit queued. Will execute and run tests."

                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })

                # If we got an edit_file call, break out of tool loop
                if edit_action is not None:
                    break

                # Nudge: after 5 tool iterations with no edit, tell agent to stop researching
                if tool_loop_iterations >= 5 and not nudge_sent:
                    nudge_sent = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have gathered enough information about the API mappings. "
                            "Stop calling get_equivalent_func and get_func_diff. "
                            "Call edit_file NOW with the complete migrated code for "
                            f"`{file_path}`. Write the entire file."
                        )
                    })

            # ---- PHASE 2: Execute the edit (force if cap hit) ----
            if edit_action is None:
                # Cap hit — force a direct code generation call (no tools)
                print(f"[WARN] Tool loop cap hit for {file_path}, forcing direct generation", file=sys.stderr)
                try:
                    with open(target_file_path, "r") as _f:
                        current_content = _f.read()
                    force_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Write the complete migrated version of `{file_path}` "
                            f"from `{state.old_lib}` to `{state.new_lib}`. "
                            f"Output ONLY the raw Python file, no markdown fences, no explanations.\n\n"
                            f"Current file:\n{current_content}"
                        )},
                    ]
                    force_resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=force_messages,
                        max_tokens=8192,
                        temperature=0.1,
                    )
                    forced_code = force_resp.choices[0].message.content or ""
                    forced_code = forced_code.strip()
                    for fence in ("```python", "```"):
                        if forced_code.startswith(fence):
                            forced_code = forced_code[len(fence):].strip()
                    if forced_code.endswith("```"):
                        forced_code = forced_code[:-3].strip()
                    if len(forced_code) > 50:
                        edit_action = MigrationAction(file_path=file_path, new_code=forced_code)
                except Exception as e:
                    print(f"[ERROR] Forced generation failed: {e}", file=sys.stderr)

            if edit_action is None:
                print(f"[WARN] No edit action produced for {file_path}, skipping", file=sys.stderr)
                break

            step_num += 1
            attempts_for_file += 1

            obs, reward, done, info = env.step(edit_action)
            final_score = reward

            log_step(step_num, f"edit:{edit_action.file_path}", reward)

            test_result = info.get("test_result", {})
            print(
                f"[INFO] Step {step_num}: status={obs.status} reward={reward} "
                f"tests={test_result.get('passed',0)}/{test_result.get('total',0)} passed",
                file=sys.stderr,
            )

            if done or reward >= 0.99:
                break

            # ---- Feed test results back into conversation for retry ----
            if test_result.get("failed", 0) > 0 or test_result.get("errors", 0) > 0:
                test_output = test_result.get("output", "")[:1500]
                retry_msg = (
                    f"Tests failed after edit. {test_result['passed']}/{test_result['total']} passing.\n"
                    f"Reward: {reward}\n\n"
                    f"Test output:\n{test_output}\n\n"
                    f"Use get_func_diff again to verify your mappings, then call edit_file "
                    f"with corrected code."
                )
                messages.append({"role": "user", "content": retry_msg})
            else:
                # All tests passing for this file, move to next
                break

    # ---- Final output ----
    log_end(task_name, final_score)

    print(f"[INFO] Workdir: {env._work_dir}", file=sys.stderr)

    work_repo = getattr(env, "_work_repo", env._work_dir)
    if work_repo and os.path.isdir(work_repo):
        for root, _, fnames in os.walk(work_repo):
            for fn in fnames:
                if fn.endswith(".py") and fn != "__init__.py":
                    fpath = os.path.join(root, fn)
                    rel = os.path.relpath(fpath, work_repo)
                    print(f"\n[OUTPUT] === {rel} ===", file=sys.stderr)
                    with open(fpath) as f:
                        print(f.read(), file=sys.stderr)

    if not KEEP_WORKDIR:
        env.cleanup()
    else:
        print(f"[INFO] Workdir kept at: {env._work_dir}", file=sys.stderr)

    print(f"[INFO] Done. Final score: {final_score}", file=sys.stderr)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Don't delete the working directory after run")
    args = parser.parse_args()
    KEEP_WORKDIR = args.keep_workdir
    run_agent()
