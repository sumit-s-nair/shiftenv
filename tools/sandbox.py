import subprocess
import importlib
import importlib.util
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