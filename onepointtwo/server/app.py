from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import ast
import uuid
import threading
from tasks import TASKS 
import uvicorn

# --- Models ---

class CodeMigrationAction(BaseModel):
    func_id: str
    rewritten_code: str

class CodeMigrationObservation(BaseModel):
    pending_functions: List[Dict[str, Any]]
    completed_functions: List[str]
    target_library: str
    last_error: Optional[str] = None

class Reward(BaseModel):
    value: float

# --- Environment ---

class CodeMigrationEnv:
    def __init__(self, task_config: Dict):
        self.config = task_config
        self.pending = {f["func_id"]: f for f in task_config["functions"]}
        self.completed = []
        self.last_error = None
        self.cumulative_reward = 0.0

    def _uses_old_library(self, code_str: str) -> bool:
        source_prefix = self.config["source_lib"].split('.')[0]
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [n.name for n in node.names]
                    if any(name.startswith(source_prefix) for name in names): return True
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(source_prefix): return True
            return source_prefix in code_str 
        except: return True

    def calculate_complex_reward(self, old_code: str, new_code: str) -> float:
        import ast

        target_lib = self.config["target_lib"]
        source_lib = self.config["source_lib"]
        reward = 0.0

        # --- Weights (tunable) ---
        W = {
            "migration": 0.25,
            "target_usage": 0.15,
            "semantic_similarity": 0.20,
            "signature": 0.10,
            "async_correctness": 0.10,
            "error_handling": 0.05,
            "code_quality": 0.10,
            "penalty_complexity": -0.10
        }

        try:
            old_tree = ast.parse(old_code)
            new_tree = ast.parse(new_code)

            # ----------------------------
            # 1. Migration correctness
            # ----------------------------
            if not self._uses_old_library(new_code):
                reward += W["migration"]
            else:
                self.last_error = f"Still uses {source_lib}"
                return -0.3

            # ----------------------------
            # 2. Target library usage
            # ----------------------------
            target_prefix = target_lib.split('.')[0]
            uses_target = any(
                isinstance(n, (ast.Import, ast.ImportFrom)) and
                (
                    any(name.name.startswith(target_prefix) for name in getattr(n, "names", [])) or
                    (hasattr(n, "module") and n.module and n.module.startswith(target_prefix))
                )
                for n in ast.walk(new_tree)
            )

            if uses_target:
                reward += W["target_usage"]
            else:
                reward -= 0.05  # weak penalty

            # ----------------------------
            # 3. Semantic similarity (AST overlap heuristic)
            # ----------------------------
            def get_node_types(tree):
                return set(type(n).__name__ for n in ast.walk(tree))

            old_nodes = get_node_types(old_tree)
            new_nodes = get_node_types(new_tree)

            overlap = len(old_nodes & new_nodes) / max(1, len(old_nodes))
            reward += W["semantic_similarity"] * overlap

            # ----------------------------
            # 4. Function signature preservation
            # ----------------------------
            def get_func_sig(tree):
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        return len(node.args.args)
                return 0

            if get_func_sig(old_tree) == get_func_sig(new_tree):
                reward += W["signature"]
            else:
                reward -= 0.05

            # ----------------------------
            # 5. Async correctness (graded, not binary)
            # ----------------------------
            is_async_target = "async" in target_lib.lower()

            async_count = sum(isinstance(n, ast.Await) for n in ast.walk(new_tree))
            async_def = any(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(new_tree))

            if is_async_target:
                if async_def:
                    reward += W["async_correctness"] * 0.6
                if async_count > 0:
                    reward += W["async_correctness"] * 0.4
            else:
                if async_def or async_count > 0:
                    reward -= 0.05  # unnecessary async usage

            # ----------------------------
            # 6. Error handling improvement
            # ----------------------------
            has_try = any(isinstance(n, ast.Try) for n in ast.walk(new_tree))
            if has_try:
                reward += W["error_handling"]

            # ----------------------------
            # 7. Code quality / idioms
            # ----------------------------
            quality_score = 0.0

            # context managers
            if any(isinstance(n, (ast.With, ast.AsyncWith)) for n in ast.walk(new_tree)):
                quality_score += 0.4

            # list/dict comprehensions
            if any(isinstance(n, (ast.ListComp, ast.DictComp)) for n in ast.walk(new_tree)):
                quality_score += 0.3

            # avoids excessive nesting
            max_depth = 0

            def get_depth(node, depth=0):
                nonlocal max_depth
                max_depth = max(max_depth, depth)
                for child in ast.iter_child_nodes(node):
                    get_depth(child, depth + 1)

            get_depth(new_tree)

            if max_depth < 10:
                quality_score += 0.3

            reward += W["code_quality"] * quality_score

            # ----------------------------
            # 8. Complexity penalty
            # ----------------------------
            def count_nodes(tree):
                return sum(1 for _ in ast.walk(tree))

            old_size = count_nodes(old_tree)
            new_size = count_nodes(new_tree)

            if new_size > old_size * 1.5:
                reward += W["penalty_complexity"]

            # ----------------------------
            # Final normalization
            # ----------------------------
            self.last_error = None
            return round(max(-1.0, min(1.0, reward)), 3)

        except SyntaxError as e:
            self.last_error = f"Syntax Error: {e}"
            return -0.6

    def step(self, action: CodeMigrationAction):
        if action.func_id not in self.pending:
            return self.state(), Reward(value=-0.1), False, {"error": "Invalid ID"}

        old_code = self.pending[action.func_id]["code"]
        reward_value = self.calculate_complex_reward(old_code, action.rewritten_code)
        
        # Move to completed only if functional success was achieved (reward > 0)
        if reward_value > 0:
            reward_value = max(0.10, min(0.90, reward_value))
            self.completed.append(action.func_id)
            del self.pending[action.func_id]
        else:
            # Penalties stay negative, which is fine
            reward_value = min(-0.01, reward_value)
        
        done = len(self.pending) == 0
        return self.state(), Reward(value=reward_value), done, {}

    def state(self) -> CodeMigrationObservation:
        return CodeMigrationObservation(
            pending_functions=list(self.pending.values()),
            completed_functions=self.completed,
            target_library=self.config["target_lib"],
            last_error=self.last_error
        )

# --- FastAPI App ---

app = FastAPI()
sessions: Dict[str, Dict] = {}
_lock = threading.Lock()

@app.get("/")
def ping():
    return {"status": "ok", "message": "space is running"}
    
@app.post("/reset")
def reset(task_id: str = "task_easy"):
    if task_id not in TASKS: raise HTTPException(404)
    env = CodeMigrationEnv(TASKS[task_id])
    sid = str(uuid.uuid4())
    with _lock: sessions[sid] = {"env": env, "rewards": []}
    return {"session_id": sid, "observation": env.state().model_dump()}

@app.post("/step")
def step(session_id: str, action: CodeMigrationAction):
    with _lock:
        if session_id not in sessions: raise HTTPException(404)
        env = sessions[session_id]["env"]
    
    obs, reward, done, info = env.step(action)
    
    with _lock: sessions[session_id]["rewards"].append(reward.value)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}

@app.get("/grader")
def grader(session_id: str):
    with _lock:
        session = sessions.get(session_id)
        if not session: raise HTTPException(404)
        
        total = sum(session["rewards"])
        n_funcs = len(TASKS[session["env"].config["id"]]["functions"])
        
        # Calculate raw average
        raw_average = total / n_funcs if n_funcs > 0 else 0.01
        

        score = max(0.10, min(0.90, raw_average))

        
    return {
        "score": float(f"{score:.3f}"), # Force 3 decimal places as a float
        "success": len(session["env"].pending) == 0
    }

def main():
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()