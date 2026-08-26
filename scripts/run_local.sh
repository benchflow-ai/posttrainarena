#!/usr/bin/env bash
# Local rollout harness — build a task package's image, run its oracle,
# score it with its verifier, and print the reward. Self-contained:
# needs only docker and bash, no benchflow install.
#
# Usage:
#   scripts/run_local.sh <task-dir>                 # oracle replay: expect reward 1.0
#   scripts/run_local.sh <task-dir> --skip-oracle   # empty trial: expect reward 0.0
#   scripts/run_local.sh <task-dir> --network       # allow network (default: none)
#
# The oracle replay proves your task is solvable; the empty trial proves
# your verifier does not hand out rewards for doing nothing. Run both
# before opening a PR. The managed pipeline remains authoritative.
set -euo pipefail

usage() { sed -n '2,12p' "$0"; exit 2; }

TASK_DIR=""
RUN_ORACLE=1
NETWORK="none"
for arg in "$@"; do
  case "$arg" in
    --skip-oracle) RUN_ORACLE=0 ;;
    --network)     NETWORK="bridge" ;;
    -h|--help)     usage ;;
    *)             TASK_DIR="$arg" ;;
  esac
done
[ -n "$TASK_DIR" ] || usage

TASK_DIR=$(cd "$TASK_DIR" && pwd)
NAME=$(basename "$TASK_DIR")
for required in environment/Dockerfile verifier/test.sh oracle/solve.sh; do
  [ -e "$TASK_DIR/$required" ] || { echo "✗ $NAME — missing $required" >&2; exit 2; }
done

IMAGE="posttrain-local/$NAME"
echo "→ building $IMAGE"
docker build -q -t "$IMAGE" "$TASK_DIR/environment"

LOGS=$(mktemp -d)
trap 'rm -rf "$LOGS"' EXIT

if [ "$RUN_ORACLE" = 1 ]; then
  TRIAL="bash /oracle/solve.sh && bash /verifier/test.sh"
  echo "→ running oracle + verifier (network: $NETWORK)"
else
  TRIAL="bash /verifier/test.sh"
  echo "→ running verifier on an empty trial (network: $NETWORK)"
fi

# --- sandbox the untrusted run ------------------------------------------
# The oracle and verifier are submitter-authored code we are about to run.
# Enforce the env's OWN declared resource budget as a hard ceiling, drop all
# Linux capabilities, forbid privilege escalation, disable swap, and cap
# wall-clock time. The run phase has no network by default (NETWORK=none);
# mounts are read-only except the throwaway /logs dir.
#
# NOTE: `docker build` above still runs the submitter's Dockerfile with build
# network (legitimate envs pip-install verifier deps at build time) — that is a
# residual trust surface handled maintainer-side, out of scope for this script.
read_fm() {  # read_fm <section> <key>  -> value from task.md frontmatter (no PyYAML)
  python3 - "$TASK_DIR/task.md" "$1" "$2" 2>/dev/null <<'PY'
import re, sys
path, section, key = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    txt = open(path, encoding="utf-8").read()
except OSError:
    sys.exit(0)
m = re.match(r"\A---\s*\n(.*?)\n---", txt, re.DOTALL)
fm = m.group(1) if m else ""
in_sec = False
for line in fm.splitlines():
    if re.match(rf"^{re.escape(section)}:\s*$", line):
        in_sec = True
        continue
    if in_sec:
        if line and not line.startswith((" ", "\t")):
            break
        mm = re.match(rf"\s+{re.escape(key)}:\s*(\S+)", line)
        if mm:
            print(mm.group(1))
            break
PY
}

MEM_MB=$(read_fm environment memory_mb);  MEM_MB=${MEM_MB:-2048}
CPUS=$(read_fm environment cpus);          CPUS=${CPUS:-2}
AGENT_T=$(read_fm agent timeout_sec);      AGENT_T=${AGENT_T:-900}
VERIF_T=$(read_fm verifier timeout_sec);   VERIF_T=${VERIF_T:-300}
# integer guards (fall back to defaults if the frontmatter value isn't numeric)
[[ "$MEM_MB" =~ ^[0-9]+$ ]] || MEM_MB=2048
[[ "$CPUS"   =~ ^[0-9]+$ ]] || CPUS=2
[[ "$AGENT_T" =~ ^[0-9]+$ ]] || AGENT_T=900
[[ "$VERIF_T" =~ ^[0-9]+$ ]] || VERIF_T=300
WALL=$(( AGENT_T + VERIF_T + 120 ))

SANDBOX=(
  --rm
  --network "$NETWORK"
  --memory "${MEM_MB}m" --memory-swap "${MEM_MB}m"
  --cpus "$CPUS"
  --pids-limit 1024
  --security-opt no-new-privileges
  --cap-drop ALL
)
echo "→ sandbox: mem ${MEM_MB}m, cpus ${CPUS}, pids 1024, cap-drop ALL, no-new-privileges, wall ${WALL}s"

# Wall-clock kill switch. GNU timeout is `timeout` on Linux/CI and `gtimeout`
# from coreutils on macOS; degrade gracefully if neither is present.
TIMEOUT_BIN=$(command -v timeout || command -v gtimeout || true)
RUN=(docker run "${SANDBOX[@]}"
  -v "$TASK_DIR/verifier":/verifier:ro
  -v "$TASK_DIR/oracle":/oracle:ro
  -v "$LOGS":/logs
  "$IMAGE" bash -c "$TRIAL")
if [ -n "$TIMEOUT_BIN" ]; then
  "$TIMEOUT_BIN" --signal=KILL "${WALL}s" "${RUN[@]}" || true
else
  echo "  (no timeout/gtimeout found — running without a wall-clock cap)" >&2
  "${RUN[@]}" || true
fi

REWARD_FILE="$LOGS/verifier/reward.txt"
if [ ! -f "$REWARD_FILE" ]; then
  echo "✗ $NAME — verifier wrote no reward to /logs/verifier/reward.txt" >&2
  exit 1
fi
REWARD=$(tr -d '[:space:]' < "$REWARD_FILE")
echo "reward: $REWARD"

if [ "$RUN_ORACLE" = 1 ]; then
  [ "$REWARD" = "1.0" ] && { echo "✓ $NAME — oracle scores 1.0"; exit 0; }
  echo "✗ $NAME — oracle replay scored $REWARD (expected 1.0)" >&2; exit 1
else
  [ "$REWARD" = "1.0" ] && { echo "✗ $NAME — empty trial scored 1.0: verifier is too weak" >&2; exit 1; }
  echo "✓ $NAME — empty trial correctly scores $REWARD"; exit 0
fi
