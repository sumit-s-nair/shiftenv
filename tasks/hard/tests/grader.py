import ast
import os


def _clamp_score(raw_score: float) -> float:
    score = max(0.01, min(0.99, raw_score))
    return score


def _score_from_trajectory(trajectory: dict | None) -> float | None:
    if not isinstance(trajectory, dict):
        return None
    rewards = trajectory.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        return _clamp_score(0.5)

    numeric_rewards = []
    for r in rewards:
        try:
            numeric_rewards.append(float(r))
        except (TypeError, ValueError):
            continue

    if not numeric_rewards:
        return _clamp_score(0.5)

    raw_score = sum(numeric_rewards) / len(numeric_rewards)
    return _clamp_score(raw_score)


def _get_imports(filepath: str) -> set[str]:
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def grade(repo_dir: str | None = None, trajectory: dict | None = None) -> float:
    """Simple hard-task grader with strict (0, 1) score bounds."""
    trajectory_score = _score_from_trajectory(trajectory)
    if trajectory_score is not None:
        return round(trajectory_score, 4)

    if isinstance(repo_dir, dict):
        trajectory_score = _score_from_trajectory(repo_dir)
        if trajectory_score is not None:
            return round(trajectory_score, 4)
        repo_dir = None

    repo_dir = repo_dir or os.path.join(os.path.dirname(__file__), "..", "repo")
    files = ["model.py", "train.py", "data.py", "predict.py"]

    checks: list[bool] = []
    for rel_path in files:
        fpath = os.path.join(repo_dir, rel_path)
        imports = _get_imports(fpath)
        checks.append("tensorflow" not in imports)
        checks.append("torch" in imports)

    passed = sum(1 for ok in checks if ok)
    total = len(checks)
    raw_score = (passed / total) if total > 0 else 0.0
    score = _clamp_score(raw_score)
    return round(score, 4)


def grader(trajectory: dict | None = None, repo_dir: str | None = None) -> float:
    return grade(repo_dir=repo_dir, trajectory=trajectory)
