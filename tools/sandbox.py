import subprocess
import importlib
import importlib.util
import re
import sys
from typing import Any

# block system-level and dangerous packages only, list more if you want
BLOCKED_LIBS = {
    "os", "sys", "subprocess", "shutil",     # system access stuff
    "ctypes", "cffi",                        # native code execution
    "socket", "paramiko",                    # raw network/ssh access 
    "scapy",                                 # packet manipulation
}


def is_installed(lib: str) -> bool:
    """Check if a library is importable in the current environment."""
    return importlib.util.find_spec(lib) is not None


def pip_install(lib: str) -> dict[str, Any]:
    """
    Install any PyPI library the agent needs, except blocked system packages.
    Decision to call this belongs to the agent, not the tools.
    """
    if lib in BLOCKED_LIBS:
        return {
            "lib": lib,
            "success": False,
            "already_installed": False,
            "output": f"'{lib}' is blocked — system or unsafe package"
        }

    if is_installed(lib):
        return {
            "lib": lib,
            "success": True,
            "already_installed": True,
            "output": f"'{lib}' is already installed"
        }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", lib],
            capture_output=True,
            text=True,
            timeout=60
        )
        importlib.invalidate_caches()
        return {
            "lib": lib,
            "success": result.returncode == 0,
            "already_installed": False,
            "output": result.stdout if result.returncode == 0 else result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "lib": lib,
            "success": False,
            "already_installed": False,
            "output": "pip install timed out after 60s"
        }
    except Exception as e:
        return {
            "lib": lib,
            "success": False,
            "already_installed": False,
            "output": str(e)
        }


def run_tests(test_dir: str, repo_dir: str | None = None) -> dict[str, Any]:
    """
    Run pytest on the given test directory/file.

    Args:
        test_dir: Path to the test directory or file to run.
        repo_dir: Working directory for pytest (defaults to test_dir parent).

    Returns:
        Dict with keys: passed, failed, errors, total, score (0.0-1.0), output.
    """
    cwd = repo_dir or test_dir
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_dir, "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
        stdout = result.stdout
        stderr = result.stderr

        # Parse pytest summary line, e.g.: "3 passed, 1 failed, 1 error"
        passed_match = re.search(r"(\d+) passed", stdout)
        failed_match = re.search(r"(\d+) failed", stdout)
        error_match = re.search(r"(\d+) error", stdout)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0

        total = passed + failed + errors
        score = passed / total if total > 0 else 0.0

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "score": round(score, 4),
            "output": stdout + stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "total": 1,
            "score": 0.0,
            "output": "pytest timed out after 120s",
        }
    except Exception as e:
        return {
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "total": 1,
            "score": 0.0,
            "output": str(e),
        }