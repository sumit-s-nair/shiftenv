import random

def generate_migration_task(difficulty: str = "easy", seed: int = 42):
    """
    Generates a migration task configuration.
    
    Easy:   Migration between similar APIs (requests -> httpx). 2 functions.
    Medium: Migration involving structural changes (sqlite3 -> SQLAlchemy Core). 4 functions.
    Hard:   Migration involving paradigm shifts (Sync requests -> Async httpx with context managers). 6 functions.
    """
    random.seed(seed)
    
    if difficulty == "easy":
        source_lib = "requests"
        target_lib = "httpx"
        n_funcs = 2
        template = "def {name}(url):\n    import requests\n    return requests.{method}(url).json()"
        methods = ["get", "post"]
    elif difficulty == "medium":
        source_lib = "sqlite3"
        target_lib = "sqlalchemy"
        n_funcs = 4
        template = "def {name}(db_path, query):\n    import sqlite3\n    conn = sqlite3.connect(db_path)\n    cur = conn.cursor()\n    cur.execute(query)\n    return cur.fetchall()"
        methods = ["fetch_all", "get_results", "query_db", "run_sql"]
    else: # Hard
        source_lib = "requests"
        target_lib = "httpx (Async)"
        n_funcs = 6
        template = "def {name}(url, data=None):\n    import requests\n    resp = requests.post(url, json=data)\n    if resp.status_code == 200:\n        return resp.text\n    return None"
        methods = ["upload", "sync_data", "backup", "telemetry", "notify", "patch_update"]

    functions = []
    for i in range(n_funcs):
        name = methods[i] if i < len(methods) else f"func_{i}"
        code = template.format(name=name, method=random.choice(["get", "post"]) if difficulty == "easy" else "")
        
        functions.append({
            "func_id": f"module_{difficulty}.py:{name}",
            "code": code,
            "library_functions_used": [f"{source_lib}.{m}" for m in (["get", "post"] if difficulty=="easy" else ["connect", "execute"])]
        })

    return {
        "id": f"task_{difficulty}",
        "difficulty": difficulty,
        "source_lib": source_lib,
        "target_lib": target_lib,
        "functions": functions,
        "max_steps": n_funcs + 2
    }

TASKS = {
    "task_easy": generate_migration_task("easy", seed=101),
    "task_medium": generate_migration_task("medium", seed=102),
    "task_hard": generate_migration_task("hard", seed=103)
}