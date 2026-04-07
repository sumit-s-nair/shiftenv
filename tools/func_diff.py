# tools/func_diff.py

import inspect
import importlib
import importlib.metadata
from typing import Any


def get_package_version(lib: str) -> str:
    """Get installed version without importing the library."""
    try:
        return importlib.metadata.version(lib)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def get_func_info(lib: str, func_path: str) -> dict[str, Any]:
    """Get signature and docstring for a function in an installed library."""
    try:
        mod = importlib.import_module(lib)
        obj = mod
        for part in func_path.split("."):
            obj = getattr(obj, part)
        return {
            "lib": lib,
            "func": func_path,
            "signature": str(inspect.signature(obj)),
            "docstring": inspect.getdoc(obj) or "No docstring available",
            "version": get_package_version(lib),
            "source": "inspect",
            "found": True
        }
    except ImportError:
        return {
            "lib": lib,
            "func": func_path,
            "signature": None,
            "docstring": None,
            "version": "not installed",
            "source": "not_found",
            "found": False,
            "error": f"'{lib}' is not installed — call pip_install('{lib}') first"
        }
    except AttributeError:
        return {
            "lib": lib,
            "func": func_path,
            "signature": None,
            "docstring": None,
            "version": get_package_version(lib),
            "source": "not_found",
            "found": False,
            "error": f"'{lib}' is installed but '{func_path}' not found — check mappings.py"
        }


def _strip_lib_prefix(lib: str, func_path: str) -> str:
    """
    Strip the library prefix from a function path if the agent included it.
    e.g. "httpx.get" with lib="httpx" → "get"
         "httpx.Client.get" with lib="httpx" → "Client.get"
         "nn.Linear" with lib="torch" → "nn.Linear" (no prefix match)
    """
    prefix = lib + "."
    if func_path.startswith(prefix):
        return func_path[len(prefix):]
    return func_path


def get_func_diff(old_lib: str, new_lib: str, func_path: str, new_func_path: str | None = None) -> dict[str, Any]:
    """
    Compare a function between old and new library.
    new_func_path handles renames e.g. requests.Session -> httpx.Client

    Automatically strips library prefix from paths so the agent can pass
    either "httpx.get" or just "get" — both work correctly.
    """
    clean_old_func = _strip_lib_prefix(old_lib, func_path)
    clean_new_func = _strip_lib_prefix(new_lib, new_func_path) if new_func_path else clean_old_func

    old_info = get_func_info(old_lib, clean_old_func)
    new_info = get_func_info(new_lib, clean_new_func)

    if not new_info["found"]:
        new_info["note"] = f"'{clean_new_func}' not found in {new_lib} — may be a response method (not top-level), or renamed. Use it as response.{clean_new_func}() in your code."

    return {
        "old": old_info,
        "new": new_info
    }


def format_diff_for_agent(diff: dict[str, Any]) -> str:
    """Format the diff as clean text the LLM agent can reason about."""
    old = diff["old"]
    new = diff["new"]

    lines = []
    lines.append(f"## Migration: {old['lib']}.{old['func']} → {new['lib']}.{new['func']}")
    lines.append(f"Versions — {old['lib']}: {old.get('version', '?')}  |  {new['lib']}: {new.get('version', '?')}")
    lines.append("")

    for label, info in [("Old", old), ("New", new)]:
        source_label = {
            "inspect": "✓ live inspect",
            "not_found": "✗ not found"
        }.get(info.get("source", ""), "?")

        lines.append(f"### {label} ({info['lib']}.{info['func']}) [{source_label}]")
        if not info["found"]:
            lines.append(f"  {info.get('error', 'Unknown error')}")
        else:
            lines.append(f"  Signature: {info['signature']}")
            lines.append(f"  Docstring: {info['docstring']}")
        lines.append("")

    if "note" in new:
        lines.append(f"### ⚠ Note\n{new['note']}")

    return "\n".join(lines)


if __name__ == "__main__":
    diff = get_func_diff("requests", "httpx", "get")
    print(format_diff_for_agent(diff))