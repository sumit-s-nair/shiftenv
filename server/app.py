# env.py — OpenEnv-compliant MigrationEnv
# Uses Pydantic BaseModel for typed actions/observations/state.


import json
import os
import re
import shutil
import uuid
from typing import Tuple, Dict, Any
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field

from tools.sandbox import run_tests




class MigrationAction(BaseModel):
    """An action the agent takes: write new code to a file."""
    file_path: str = Field(description="Relative path to the file to edit within the repo")
    new_code: str = Field(description="Complete new file contents to write")


class MigrationObservation(BaseModel):
    """What the agent sees after each step."""
    status: str = Field(description="One of: start, continue, success, failed, error")
    message: str = Field(description="Human-readable status message")
    context: str = Field(default="", description="File contents, API specs, or error details")
    reward: float = Field(default=0.0, description="Current reward signal")


class MigrationState(BaseModel):
    """Episode metadata."""
    episode_id: str = Field(description="Unique episode identifier")
    task_name: str = Field(default="", description="Name of the current task (easy/medium/hard)")
    old_lib: str = Field(default="", description="Library being migrated from")
    new_lib: str = Field(default="", description="Library being migrated to")
    current_step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=10, description="Maximum allowed steps")
    is_done: bool = Field(default=False, description="Whether the episode has ended")
    files_in_repo: list[str] = Field(default_factory=list, description="Python files in the repo")


# ---------------------------------------------------------
# 2. Task Configuration
# ---------------------------------------------------------
TASK_CONFIG = {
    "easy": {
        "old_lib": "random",
        "new_lib": "secrets",
        "repo_subdir": "tasks/easy/repo",
        "test_subdir": "tasks/easy/tests",
        "description": "Migrate crypto utilities from `random` to `secrets`.",
    },
    "medium": {
        "old_lib": "requests",
        "new_lib": "httpx",
        "repo_subdir": "tasks/medium/repo",
        "test_subdir": "tasks/medium/tests",
        "description": "Migrate HTTP client from `requests` to `httpx`.",
    },
    "hard": {
        "old_lib": "tensorflow",
        "new_lib": "torch",
        "repo_subdir": "tasks/hard/repo",
        "test_subdir": "tasks/hard/tests",
        "description": "Migrate ML pipeline from TensorFlow/Keras to PyTorch.",
    },
}


# ---------------------------------------------------------
# 3. Reward Helpers
# ---------------------------------------------------------
def _clamp_score(raw_score: float) -> float:
    # Keep reward components strictly inside (0, 1).
    score = max(0.01, min(0.99, raw_score))
    return score


def _check_imports(repo_path: str, old_lib: str) -> float:
    """
    Returns a clamped score near 1.0 if old imports are gone.
    Returns a clamped score near 0.0 if any old import remains.
    """
    for root, _, files in os.walk(repo_path):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
                if re.search(rf"\bimport\s+{re.escape(old_lib)}\b|from\s+{re.escape(old_lib)}\b", content):
                    return _clamp_score(0.0)
            except Exception:
                continue
    return _clamp_score(1.0)


def _compute_reward(test_score: float, repo_path: str, old_lib: str,
                    current_step: int, max_steps: int) -> float:
    """
    Weighted reward with partial progress signal.
    - 0.7 weight on test pass rate (ground truth)
    - 0.3 weight on import cleanliness (necessary but not sufficient)
    - small step penalty to discourage aimless actions
    """
    import_score = _check_imports(repo_path, old_lib)
    step_penalty = 0.01 * (current_step / max_steps) if max_steps > 0 else 0
    raw_score = 0.7 * test_score + 0.3 * import_score - step_penalty
    score = _clamp_score(raw_score)
    return round(score, 4)


# ---------------------------------------------------------
# 4. The OpenEnv Environment
# ---------------------------------------------------------
class MigrationEnv:
    """
    RL environment for library migration tasks.

    The agent is given a Python repo that uses an outdated library and must
    migrate it to a target library. The reward signal comes from running
    the test suite (the grader).

    Workdir layout (mirrors task directory so relative paths in graders work):
        .workdir/<episode_id>/
            repo/       ← working copy of the source code (agent edits here)
            tests/      ← copy of the grader tests (reads ../repo/ via __file__)
    """

    def __init__(self, base_dir: str | None = None, max_steps: int = 10):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.max_steps = max_steps
        self._episode_id = ""
        self._task_name = ""
        self._old_lib = ""
        self._new_lib = ""
        self._work_dir = ""      # top-level workdir
        self._work_repo = ""     # workdir/repo/   (agent writes here)
        self._work_tests = ""    # workdir/tests/  (grader runs from here)
        self._current_step = 0
        self._done = False
        self._last_reward = 0.0

    @property
    def state(self) -> MigrationState:
        """Return current episode state."""
        files = []
        if self._work_repo and os.path.isdir(self._work_repo):
            for root, _, fnames in os.walk(self._work_repo):
                for fn in fnames:
                    if fn.endswith(".py"):
                        files.append(os.path.relpath(os.path.join(root, fn), self._work_repo))
        return MigrationState(
            episode_id=self._episode_id,
            task_name=self._task_name,
            old_lib=self._old_lib,
            new_lib=self._new_lib,
            current_step=self._current_step,
            max_steps=self.max_steps,
            is_done=self._done,
            files_in_repo=sorted(files),
        )

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, task_name: str | None = None) -> MigrationObservation:
        """
        Initialize a new migration episode.

        Args:
            task_name: One of 'easy', 'medium', 'hard'.
                       Defaults to OPENENV_TASK env var or 'medium'.
        """
        task_name = task_name or os.environ.get("OPENENV_TASK", "medium")
        if task_name not in TASK_CONFIG:
            return MigrationObservation(
                status="error",
                message=f"Unknown task '{task_name}'. Choose from: {list(TASK_CONFIG.keys())}",
            )

        config = TASK_CONFIG[task_name]
        self._episode_id = str(uuid.uuid4())[:8]
        self._task_name = task_name
        self._old_lib = config["old_lib"]
        self._new_lib = config["new_lib"]
        self._current_step = 0
        self._done = False

        # Resolve original paths
        orig_repo = os.path.join(self.base_dir, config["repo_subdir"])
        orig_tests = os.path.join(self.base_dir, config["test_subdir"])

        # Create workdir mirroring the task layout:
        #   .workdir/<id>/repo/   ← agent edits here
        #   .workdir/<id>/tests/  ← grader reads ../repo/ via __file__
        self._work_dir = os.path.join(self.base_dir, ".workdir", self._episode_id)
        self._work_repo = os.path.join(self._work_dir, "repo")
        self._work_tests = os.path.join(self._work_dir, "tests")

        if os.path.exists(self._work_dir):
            shutil.rmtree(self._work_dir)

        shutil.copytree(orig_repo, self._work_repo)
        shutil.copytree(orig_tests, self._work_tests)

        # Read all source files for context
        file_contents = {}
        for root, _, fnames in os.walk(self._work_repo):
            for fn in fnames:
                if fn.endswith(".py") and fn != "__init__.py":
                    fpath = os.path.join(root, fn)
                    rel = os.path.relpath(fpath, self._work_repo)
                    with open(fpath, "r") as f:
                        file_contents[rel] = f.read()

        # Run initial tests to establish baseline reward
        test_result = run_tests(self._work_tests, repo_dir=self._work_dir)
        initial_reward = _compute_reward(
            test_result["score"], self._work_repo, self._old_lib, 0, self.max_steps
        )
        self._last_reward = initial_reward

        # Build context string
        context_parts = [
            f"=== TASK: {config['description']} ===",
            f"Migrate from `{self._old_lib}` to `{self._new_lib}`",
            f"Max steps: {self.max_steps}",
            f"Initial test score: {test_result['score']} ({test_result['passed']}/{test_result['total']} passing)",
            f"Initial reward: {initial_reward}",
            "",
        ]
        for rel_path, code in sorted(file_contents.items()):
            context_parts.append(f"--- {rel_path} ---")
            context_parts.append(code)
            context_parts.append("")

        return MigrationObservation(
            status="start",
            message=config["description"],
            context="\n".join(context_parts),
            reward=initial_reward,
        )

    def step(self, action: MigrationAction) -> Tuple[MigrationObservation, float, bool, Dict[str, Any]]:
        """
        Execute an action: write new code and evaluate.

        Returns: (observation, reward, done, info)
        """
        if self._done:
            return (
                MigrationObservation(status="error", message="Episode already done. Call reset()."),
                0.0, True, {"step": self._current_step}
            )

        self._current_step += 1

        # Write the new code to the REPO subdirectory of the workdir
        target_path = os.path.join(self._work_repo, action.file_path)
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w") as f:
                f.write(action.new_code)
        except Exception as e:
            return (
                MigrationObservation(
                    status="error",
                    message=f"Failed to write file: {action.file_path}",
                    context=str(e),
                    reward=-0.1,
                ),
                -0.1, False, {"step": self._current_step}
            )

        # Run tests from the TESTS subdirectory (grader reads ../repo/ via __file__)
        test_result = run_tests(self._work_tests, repo_dir=self._work_dir)
        reward = _compute_reward(
            test_result["score"], self._work_repo, self._old_lib,
            self._current_step, self.max_steps
        )
        self._last_reward = reward

        # Check completion conditions
        if reward >= 0.99:
            self._done = True
            return (
                MigrationObservation(
                    status="success",
                    message=f"Migration complete! All tests passing. Score: {reward}",
                    reward=reward,
                ),
                reward, True, {"step": self._current_step, "test_result": test_result}
            )

        if self._current_step >= self.max_steps:
            self._done = True
            return (
                MigrationObservation(
                    status="failed",
                    message=f"Max steps ({self.max_steps}) reached. Final reward: {reward}",
                    context=f"Test output:\n{test_result['output'][:500]}",
                    reward=reward,
                ),
                reward, True, {"step": self._current_step, "test_result": test_result}
            )

        # Read back the current file to show the agent what it wrote
        try:
            with open(target_path, "r") as f:
                current_code = f.read()
        except Exception:
            current_code = ""

        return (
            MigrationObservation(
                status="continue",
                message=(
                    f"Step {self._current_step}/{self.max_steps}. "
                    f"Tests: {test_result['passed']}/{test_result['total']} passing. "
                    f"Reward: {reward}"
                ),
                context=(
                    f"--- {action.file_path} (current) ---\n{current_code}\n\n"
                    f"--- Test Output ---\n{test_result['output'][:1000]}"
                ),
                reward=reward,
            ),
            reward, False, {"step": self._current_step, "test_result": test_result}
        )

    def cleanup(self):
        """Remove the working directory."""
        if self._work_dir and os.path.exists(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)


app = Flask(__name__)

# Instantiate a single global environment for the container runtime
env = MigrationEnv()

@app.route("/", methods=["GET", "POST"])
def ping():
    """
    Automated ping to the Space URL.
    Must return 200 OK.
    """
    return jsonify({"status": "ok", "message": "ShiftEnv space is running"}), 200

@app.route("/reset", methods=["POST"])
def reset():
    """
    OpenEnv /reset endpoint.
    Accepts an optional JSON payload to specify the task_name.
    """
    data = request.get_json(silent=True) or {}
    task_name = data.get("task_name")
    
    # Call the underlying environment reset
    obs = env.reset(task_name=task_name)
    
    # Pydantic v2 uses model_dump(), for v1 use .dict()
    return jsonify(obs.model_dump()), 200

@app.route("/step", methods=["POST"])
def step():
    """
    OpenEnv /step endpoint.
    Accepts a JSON payload matching the MigrationAction schema.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    try:
        # Validate incoming data using the Pydantic model
        action = MigrationAction(**data)
    except Exception as e:
        return jsonify({"error": "Invalid action format", "details": str(e)}), 400

    # Execute the step
    obs, reward, done, info = env.step(action)
    
    # Return the combined result expected by OpenEnv
    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }), 200

@app.route("/state", methods=["GET"])
def state():
    """
    OpenEnv /state endpoint.
    Returns the current episode metadata.
    """
    current_state = env.state
    return jsonify(current_state.model_dump()), 200

def main():
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()