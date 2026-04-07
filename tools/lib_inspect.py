# tools/lib_inspector.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any

# Load .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Static known-good mappings — checked BEFORE the LLM to avoid hallucination.
# Format: (old_lib, old_func, new_lib) -> (new_func, note)
# ---------------------------------------------------------------------------
KNOWN_MAPPINGS: dict[tuple[str, str, str], tuple[str, str]] = {
    # requests → httpx (synchronous equivalents)
    ("requests", "get",         "httpx"): ("get",         "httpx.get is synchronous. Use follow_redirects= instead of allow_redirects=."),
    ("requests", "post",        "httpx"): ("post",        "httpx.post is synchronous. Same signature, no behavior change for basic use."),
    ("requests", "put",         "httpx"): ("put",         "httpx.put is synchronous."),
    ("requests", "patch",       "httpx"): ("patch",       "httpx.patch is synchronous."),
    ("requests", "delete",      "httpx"): ("delete",      "httpx.delete is synchronous."),
    ("requests", "head",        "httpx"): ("head",        "httpx.head is synchronous. Use follow_redirects=True instead of allow_redirects=True."),
    ("requests", "Session",     "httpx"): ("Client",      "httpx.Client is the sync equivalent of requests.Session. For async use httpx.AsyncClient."),
    ("requests", "Response",    "httpx"): ("Response",    "Same concept, same method names (raise_for_status, json, text, content)."),

    # requests adapter/retry → httpx transport
    ("requests.adapters", "HTTPAdapter", "httpx"): ("HTTPTransport", "httpx.HTTPTransport replaces HTTPAdapter. Pass retries=N directly."),
    ("urllib3.util.retry", "Retry",      "httpx"): ("HTTPTransport", "Use httpx.HTTPTransport(retries=N) — no separate Retry class in httpx."),

    # timeout
    ("requests", "Timeout",     "httpx"): ("Timeout",     "httpx.Timeout(connect=C, read=R) replaces tuple (connect, read). Required for explicit control."),

    # random → secrets
    ("random", "choice",        "secrets"): ("choice",       "secrets.choice(seq) — identical signature, cryptographically secure."),
    ("random", "choices",       "secrets"): ("token_hex",    "For hex tokens use secrets.token_hex(n_bytes). For charset use [secrets.choice(charset) for _ in range(length)]."),
    ("random", "randint",       "secrets"): ("randbelow",    "secrets.randbelow(n) returns 0..n-1. For randint(a,b): secrets.randbelow(b-a+1)+a."),
    ("random", "getrandbits",   "secrets"): ("token_bytes",  "secrets.token_bytes(n) returns n random bytes. For hex: secrets.token_hex(n)."),
    ("random", "shuffle",       "secrets"): ("SystemRandom", "Use secrets.SystemRandom().shuffle(lst) — no direct shuffle in secrets module."),
    ("random", "random",        "secrets"): ("SystemRandom", "Use secrets.SystemRandom().random() — but prefer token_bytes for crypto use."),

    # tensorflow → torch (common layer mappings)
    ("tensorflow", "keras.layers.Dense",           "torch"): ("nn.Linear",       "nn.Linear(in_features, out_features). TF Dense only needs units; PyTorch requires both dims from the start."),
    ("tensorflow", "keras.layers.BatchNormalization", "torch"): ("nn.BatchNorm1d", "nn.BatchNorm1d(num_features). For 2D input use BatchNorm1d, for 4D images use BatchNorm2d."),
    ("tensorflow", "keras.layers.Dropout",         "torch"): ("nn.Dropout",      "nn.Dropout(p=rate). Note: TF rate is drop probability, PyTorch p is also drop probability — same semantics."),
    ("tensorflow", "keras.Model",                  "torch"): ("nn.Module",       "Subclass nn.Module. Implement forward() instead of call(). No training= arg; use model.train()/model.eval() instead."),
    ("tensorflow", "GradientTape",                 "torch"): ("autograd",        "Use loss.backward() + optimizer.step() + optimizer.zero_grad(). No tape context manager needed."),
    ("tensorflow", "keras.optimizers.Adam",        "torch"): ("optim.Adam",      "torch.optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2))."),
    ("tensorflow", "keras.losses.SparseCategoricalCrossentropy", "torch"): ("nn.CrossEntropyLoss", "nn.CrossEntropyLoss() takes raw LOGITS (not softmax output). Remove softmax from model output when using this loss."),
    ("tensorflow", "data.Dataset",                 "torch"): ("utils.data.DataLoader", "Use TensorDataset + DataLoader. .batch().shuffle() → DataLoader(batch_size=N, shuffle=True)."),
    ("tensorflow", "saved_model.save",             "torch"): ("save",            "torch.save(model.state_dict(), path). Load with model.load_state_dict(torch.load(path))."),
    ("tensorflow", "keras.models.load_model",      "torch"): ("load",            "model.load_state_dict(torch.load(path, weights_only=True))."),
}


def get_equivalent_func(old_lib: str, old_func: str, new_lib: str, failed: list[str] | None = None) -> dict[str, Any]:
    """
    Find the equivalent function/class in new_lib for a given old_lib function.
    Checks static known mappings first, then falls back to LLM.
    Pass failed=[] with previously tried names so the LLM doesn't repeat them.
    """
    # --- Static lookup first ---
    key = (old_lib, old_func, new_lib)
    if key in KNOWN_MAPPINGS:
        new_func, note = KNOWN_MAPPINGS[key]
        # Honour the failed list — if the static answer was tried, fall through to LLM
        if failed and new_func in failed:
            pass  # fall through to LLM below
        else:
            return {
                "old": f"{old_lib}.{old_func}",
                "new": new_func,
                "note": note,
                "found": True,
                "source": "static_mapping",
            }

    # --- LLM fallback ---
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
            "found": equivalent is not None,
            "source": "llm",
        }

    except Exception as e:
        return {
            "old": f"{old_lib}.{old_func}",
            "new": None,
            "note": None,
            "found": False,
            "error": str(e),
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