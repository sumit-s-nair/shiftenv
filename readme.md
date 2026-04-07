# ShiftEnv

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) RL environment where an AI agent migrates Python repositories from one library to another.

---

## How It Works

Give the agent a old library and it will rewrite it to the new one

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the agent

```bash
# Run on the medium task (default)
python inference.py

# Run on a specific task
OPENENV_TASK=easy python inference.py
OPENENV_TASK=hard python inference.py
```

## Reward Function

```python
reward = 0.7 * test_pass_rate + 0.3 * import_cleanliness - step_penalty
```

- **test_pass_rate** (0.0–1.0): Fraction of grader tests passing
- **import_cleanliness** (0.0 or 1.0): 1.0 if old library fully removed
- **step_penalty**: `0.01 × (current_step / max_steps)` — encourages efficiency

## Tools

| Tool | Purpose |
|------|---------|
| `lib_inspect.get_equivalent_func()` | Find equivalent function name in new library |
| `func_diff.get_func_diff()` | Compare signatures + docs between old/new library |
| `sandbox.pip_install()` | Install PyPI packages (blocked list enforced) |
| `sandbox.run_tests()` | Run pytest and get pass/fail scores |

## License

MIT
