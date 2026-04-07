# tools/lib_inspector.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any

# Load .env file
load_dotenv()


def get_equivalent_func(old_lib: str, old_func: str, new_lib: str, failed: list[str] | None = None) -> dict[str, Any]:
    """
    Find the equivalent function/class in new_lib for a given old_lib function.
    Pass failed=[] with previously tried names so the LLM doesn't repeat them.
    Verification and retry decisions belong to the agent via func_diff feedback.
    """
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    )

    prompt = _build_prompt(old_lib, old_func, new_lib, failed or [])

    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        raw = response.choices[0].message.content
        equivalent, note = _parse_response(raw)

        return {
            "old": f"{old_lib}.{old_func}",
            "new": equivalent,
            "note": note,
            "found": equivalent is not None
        }

    except Exception as e:
        return {
            "old": f"{old_lib}.{old_func}",
            "new": None,
            "note": None,
            "found": False,
            "error": str(e)
        }


def _build_prompt(old_lib: str, old_func: str, new_lib: str, failed: list[str]) -> str:
    prompt = f"""You are a Python migration expert.
                What is the equivalent of `{old_lib}.{old_func}` in `{new_lib}`?

                Respond in this exact format and nothing else:
                EQUIVALENT: <new_lib.equivalent_name or NONE if no equivalent exists>
                NOTE: <one line — most important BREAKING behavioral difference only, not general features. Focus on default value changes, renamed parameters, removed parameters, or async requirements. Or None if truly identical.>"""

    if failed:
        failed_str = ", ".join(failed)
        prompt += f"\n\nThese were already tried and are incorrect: {failed_str}\nDo not suggest these again."

    return prompt


def _parse_response(raw: str) -> tuple[str | None, str | None]:
    equivalent = None
    note = None

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("EQUIVALENT:"):
            value = line.replace("EQUIVALENT:", "").strip()
            equivalent = None if value.upper() == "NONE" else value
        elif line.startswith("NOTE:"):
            value = line.replace("NOTE:", "").strip()
            note = None if value.lower() == "none" else value

    return equivalent, note


if __name__ == "__main__":
    import sys
    old_lib = sys.argv[1] if len(sys.argv) > 1 else "requests"
    old_func = sys.argv[2] if len(sys.argv) > 2 else "get"
    new_lib = sys.argv[3] if len(sys.argv) > 3 else "httpx"

    result = get_equivalent_func(old_lib, old_func, new_lib)

    if result["found"]:
        print(f"{result['old']} → {result['new']}")
        if result["note"]:
            print(f"⚠ {result['note']}")
        print(f"\nNext: call func_diff('{old_lib}', '{new_lib}', '{old_func}', '{result['new'].split('.')[-1]}')")
    else:
        print(f"✗ No equivalent found for {result['old']} in {new_lib}")
        if "error" in result:
            print(f"  Error: {result['error']}")