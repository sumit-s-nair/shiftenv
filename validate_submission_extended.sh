#!/usr/bin/env bash
#
# validate_submission_extended.sh
#
# Extended pre-submission validator for OpenEnv hackathon projects.
#
# Checks:
#   1) HF Space ping and /reset health
#   2) Required env vars and inference.py presence/conventions
#   3) Docker build
#   4) openenv validate
#   5) Task grader discovery and strict score range (0, 1)
#   6) inference.py run and structured log format for easy/medium/hard
#
# Usage:
#   chmod +x validate_submission_extended.sh
#   ./validate_submission_extended.sh <ping_url> [repo_dir]
#
# Optional env vars:
#   SKIP_PING=1            Skip HF ping check
#   SKIP_DOCKER=1          Skip docker build check
#   SKIP_INFERENCE=1       Skip inference execution checks
#   DOCKER_BUILD_TIMEOUT   Default: 1800
#   INFERENCE_TIMEOUT      Default: 1200 (seconds per task)
#   TASKS_TO_RUN           Default: "easy medium hard"

set -uo pipefail

DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-1800}"
INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-1200}"
TASKS_TO_RUN="${TASKS_TO_RUN:-easy medium hard}"

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    return "$rc"
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+${CLEANUP_FILES[@]}}"; }
trap cleanup EXIT

normalize_ping_url() {
  local input="$1"
  input="${input%/}"

  if [[ "$input" =~ ^https?://huggingface\.co/spaces/([^/]+)/([^/?#]+) ]]; then
    local org="${BASH_REMATCH[1]}"
    local space="${BASH_REMATCH[2]}"
    local slug
    slug="$(printf "%s-%s" "$org" "$space" | tr '[:upper:]' '[:lower:]')"
    printf "https://%s.hf.space" "$slug"
    return
  fi

  printf "%s" "$input"
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ "${SKIP_PING:-0}" != "1" ] && [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  printf "\n"
  printf "Set SKIP_PING=1 to run local-only checks without a Space URL.\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="$(normalize_ping_url "$PING_URL")"
PING_URL="${PING_URL%/}"
PASS=0
TOTAL=6

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}======================================================${NC}\n"
printf "${BOLD}  OpenEnv Extended Submission Validator${NC}\n"
printf "${BOLD}======================================================${NC}\n"
log "Repo:     $REPO_DIR"
if [ "${SKIP_PING:-0}" = "1" ]; then
  log "Ping URL: (skipped)"
else
  log "Ping URL: $PING_URL"
fi
printf "\n"

# -----------------------------------------------------------------------------
# Step 1: Ping HF Space
# -----------------------------------------------------------------------------
log "${BOLD}Step 1/6: Pinging HF Space${NC} ..."

if [ "${SKIP_PING:-0}" = "1" ]; then
  pass "HF Space ping skipped (SKIP_PING=1)"
else
  CURL_OUTPUT="$(portable_mktemp validate-curl)"
  CLEANUP_FILES+=("$CURL_OUTPUT")

  HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" -d '{}' \
    "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

  if [ "$HTTP_CODE" = "200" ]; then
    pass "HF Space is live and responds to /reset"
  elif [ "$HTTP_CODE" = "000" ]; then
    fail "HF Space not reachable (connection failed or timed out)"
    hint "Check network and Space runtime status."
    hint "Try: curl -s -o /dev/null -w '%{http_code}' -X POST $PING_URL/reset"
    stop_at "Step 1"
  else
    fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
    hint "Verify URL and that the Space is running."
    stop_at "Step 1"
  fi
fi

# -----------------------------------------------------------------------------
# Step 2: Required env vars + inference static checks
# -----------------------------------------------------------------------------
log "${BOLD}Step 2/6: Checking required env vars and inference conventions${NC} ..."

if [ ! -f "$REPO_DIR/inference.py" ]; then
  fail "Missing inference.py at repo root"
  stop_at "Step 2"
fi

missing_vars=()
for v in API_BASE_URL MODEL_NAME HF_TOKEN; do
  if [ -z "${!v:-}" ]; then
    missing_vars+=("$v")
  fi
done

if [ "${#missing_vars[@]}" -gt 0 ]; then
  fail "Required env vars not set: ${missing_vars[*]}"
  hint "Export API_BASE_URL, MODEL_NAME, and HF_TOKEN before validation."
  stop_at "Step 2"
fi

if ! grep -q "from openai import OpenAI" "$REPO_DIR/inference.py"; then
  fail "inference.py does not import OpenAI client"
  stop_at "Step 2"
fi

if ! grep -q "OpenAI(" "$REPO_DIR/inference.py"; then
  fail "inference.py does not construct OpenAI client"
  stop_at "Step 2"
fi

if ! grep -q "\[START\]" "$REPO_DIR/inference.py" || \
   ! grep -q "\[STEP\]" "$REPO_DIR/inference.py" || \
   ! grep -q "\[END\]" "$REPO_DIR/inference.py"; then
  fail "inference.py does not appear to emit mandatory structured logs"
  stop_at "Step 2"
fi

pass "Required env vars and inference conventions look correct"

# -----------------------------------------------------------------------------
# Step 3: Docker build
# -----------------------------------------------------------------------------
log "${BOLD}Step 3/6: Running docker build${NC} ..."

if [ "${SKIP_DOCKER:-0}" = "1" ]; then
  pass "Docker build skipped (SKIP_DOCKER=1)"
else
  if ! command -v docker >/dev/null 2>&1; then
    fail "docker command not found"
    hint "Install Docker: https://docs.docker.com/get-docker/"
    stop_at "Step 3"
  fi

  if [ -f "$REPO_DIR/Dockerfile" ]; then
    DOCKER_CONTEXT="$REPO_DIR"
  elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
    DOCKER_CONTEXT="$REPO_DIR/server"
  else
    fail "No Dockerfile found in repo root or server/"
    stop_at "Step 3"
  fi

  log "  Found Dockerfile in $DOCKER_CONTEXT"
  BUILD_OK=false
  BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

  if [ "$BUILD_OK" = true ]; then
    pass "Docker build succeeded"
  else
    fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
    printf "%s\n" "$BUILD_OUTPUT" | tail -20
    if printf "%s" "$BUILD_OUTPUT" | grep -qi "context canceled"; then
      hint "Build likely exceeded timeout during image export. Retry with a larger timeout, e.g. DOCKER_BUILD_TIMEOUT=2400."
    fi
    stop_at "Step 3"
  fi
fi

# -----------------------------------------------------------------------------
# Step 4: openenv validate
# -----------------------------------------------------------------------------
log "${BOLD}Step 4/6: Running openenv validate${NC} ..."

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 4"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 4"
fi

# -----------------------------------------------------------------------------
# Step 5: Grader discovery + strict score range
# -----------------------------------------------------------------------------
log "${BOLD}Step 5/6: Checking tasks and graders${NC} ..."

GRADER_CHECK_OUTPUT=$(cd "$REPO_DIR" && python - <<'PY'
import importlib
import pathlib
import re
import sys

root = pathlib.Path('.')
cfg = root / 'openenv.yaml'
if not cfg.exists():
    print('ERROR: openenv.yaml not found')
    raise SystemExit(2)

text = cfg.read_text(encoding='utf-8')
grader_specs = re.findall(r'^\s*grader:\s*["\']?([^"\'\n]+)["\']?\s*$', text, re.MULTILINE)

if len(grader_specs) < 3:
    print(f'ERROR: found {len(grader_specs)} grader entries, expected at least 3')
    raise SystemExit(3)

print(f'Found {len(grader_specs)} grader entries')

fails = 0
for spec in grader_specs:
    spec = spec.strip()
    if ':' in spec:
        mod_name, fn_name = spec.split(':', 1)
    else:
        try:
            mod_name, fn_name = spec.rsplit('.', 1)
        except ValueError:
            print(f'ERROR: invalid grader reference: {spec}')
            fails += 1
            continue

    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f'ERROR: import failed for {mod_name}: {e}')
        fails += 1
        continue

    if not hasattr(mod, fn_name):
        print(f'ERROR: {spec} missing callable {fn_name}')
        fails += 1
        continue

    fn = getattr(mod, fn_name)
    score = None
    call_errors = []

    for args in [(), ({},), ({'rewards': [0.67, 0.66, 0.66, 0.96]},)]:
        try:
            score = float(fn(*args))
            break
        except Exception as e:
            call_errors.append(str(e))

    if score is None:
        print(f'ERROR: could not execute {spec}: {call_errors[-1] if call_errors else "unknown"}')
        fails += 1
        continue

    strict = (score > 0.0 and score < 1.0 and score not in (0.0, 1.0))
    print(f'{spec} -> score={score:.4f} strict={strict}')
    if not strict:
        fails += 1

if fails:
    print(f'ERROR: {fails} grader checks failed')
    raise SystemExit(4)

print('OK: all grader checks passed')
PY
) && GRADER_OK=true || GRADER_OK=false

if [ "$GRADER_OK" = true ]; then
  pass "Found >=3 graders and all scores are strictly within (0, 1)"
  log "  $GRADER_CHECK_OUTPUT"
else
  fail "Grader/task validation failed"
  printf "%s\n" "$GRADER_CHECK_OUTPUT"
  stop_at "Step 5"
fi

# -----------------------------------------------------------------------------
# Step 6: inference.py runtime + log format checks
# -----------------------------------------------------------------------------
log "${BOLD}Step 6/6: Running inference and validating [START]/[STEP]/[END] logs${NC} ..."

if [ "${SKIP_INFERENCE:-0}" = "1" ]; then
  pass "Inference run skipped (SKIP_INFERENCE=1)"
else
  for task in $TASKS_TO_RUN; do
    log "  Running OPENENV_TASK=$task (timeout=${INFERENCE_TIMEOUT}s)"
    INF_OUT="$(portable_mktemp validate-inference-${task})"
    CLEANUP_FILES+=("$INF_OUT")

    INFER_OK=false
    run_with_timeout "$INFERENCE_TIMEOUT" bash -lc "cd '$REPO_DIR' && OPENENV_TASK='$task' python inference.py" >"$INF_OUT" 2>&1 && INFER_OK=true

    if [ "$INFER_OK" != true ]; then
      fail "inference.py failed for task=$task"
      tail -40 "$INF_OUT"
      stop_at "Step 6"
    fi

    FORMAT_OUTPUT=$(python - "$INF_OUT" "$task" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
expected_task = sys.argv[2]
lines = path.read_text(encoding='utf-8', errors='replace').splitlines()

start = [l for l in lines if l.startswith('[START] ')]
steps = [l for l in lines if l.startswith('[STEP] ')]
end = [l for l in lines if l.startswith('[END] ')]

if len(start) != 1:
    print(f'ERROR: expected 1 [START], got {len(start)}')
    raise SystemExit(2)
if len(end) != 1:
    print(f'ERROR: expected 1 [END], got {len(end)}')
    raise SystemExit(3)
if len(steps) < 1:
    print('ERROR: expected at least 1 [STEP] line')
    raise SystemExit(4)

start_re = re.compile(r'^\[START\] task=(\S+) env=(\S+) model=(.+)$')
step_re = re.compile(r'^\[STEP\] step=(\d+) action=(.*) reward=(-?\d+\.\d{2}) done=(true|false) error=(.*)$')
end_re = re.compile(r'^\[END\] success=(true|false) steps=(\d+) score=([0-9]+(?:\.[0-9]+)?) rewards=(.*)$')

m_start = start_re.match(start[0])
if not m_start:
    print('ERROR: [START] format invalid')
    raise SystemExit(5)

if m_start.group(1) != expected_task:
    print(f'ERROR: [START] task mismatch. expected={expected_task} got={m_start.group(1)}')
    raise SystemExit(6)

for i, s in enumerate(steps, start=1):
    m = step_re.match(s)
    if not m:
        print(f'ERROR: [STEP] format invalid at line {i}: {s}')
        raise SystemExit(7)

m_end = end_re.match(end[0])
if not m_end:
    print('ERROR: [END] format invalid')
    raise SystemExit(8)

steps_count = int(m_end.group(2))
score = float(m_end.group(3))
if steps_count < 1:
    print('ERROR: [END] steps must be >= 1')
    raise SystemExit(9)
if score < 0.0 or score > 1.0:
    print(f'ERROR: [END] score out of [0,1]: {score}')
    raise SystemExit(10)

print(f'OK: task={expected_task} steps={steps_count} score={score:.4f}')
PY
)

    if [ $? -ne 0 ]; then
      fail "Structured output validation failed for task=$task"
      printf "%s\n" "$FORMAT_OUTPUT"
      tail -20 "$INF_OUT"
      stop_at "Step 6"
    fi

    log "    $FORMAT_OUTPUT"
  done

  pass "inference.py completed with valid structured logs for tasks: $TASKS_TO_RUN"
fi

printf "\n"
printf "${BOLD}======================================================${NC}\n"
printf "${GREEN}${BOLD}  All ${PASS}/${TOTAL} checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Submission looks ready.${NC}\n"
printf "${BOLD}======================================================${NC}\n"
printf "\n"
