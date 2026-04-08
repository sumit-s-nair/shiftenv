import ast
import os


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


def grade(repo_dir: str | None = None) -> float:
    """Simple easy-task grader with strict (0, 1) score bounds."""
    repo_dir = repo_dir or os.path.join(os.path.dirname(__file__), "..", "repo")
    files = ["crypto_utils.py", "session.py"]

    checks: list[bool] = []
    for rel_path in files:
        fpath = os.path.join(repo_dir, rel_path)
        imports = _get_imports(fpath)
        checks.append("random" not in imports)
        checks.append("secrets" in imports)

    passed = sum(1 for ok in checks if ok)
    total = len(checks)
    raw_score = (passed / total) if total > 0 else 0.0
    score = max(0.01, min(0.99, raw_score))
    return round(score, 4)


def grader(repo_dir: str | None = None) -> float:
    return grade(repo_dir)
