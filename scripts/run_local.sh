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

docker run --rm --network "$NETWORK" \
  -v "$TASK_DIR/verifier":/verifier:ro \
  -v "$TASK_DIR/oracle":/oracle:ro \
  -v "$LOGS":/logs \
  "$IMAGE" bash -c "$TRIAL" || true

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
