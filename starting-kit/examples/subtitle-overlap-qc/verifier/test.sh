#!/bin/bash
# Verifier entry point. Standard PostTrain Arena / Benchflow shim:
# runs pytest against verifier/test_outputs.py, then writes the canonical
# reward artifacts.
#
# The task image (python:3.11-slim) bakes pytest + pytest-json-ctrf in at
# build time, so the normal path needs no network at verify time. The uvx
# bootstrap remains only as a fallback for harnesses that run the verifier
# on a different image.

VERIFIER_DIR="${BENCHFLOW_VERIFIER_DIR:-/verifier}"
if [ ! -d "$VERIFIER_DIR" ] && [ -d /tests ]; then
  VERIFIER_DIR=/tests
fi

REWARD_TEXT="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"
REWARD_JSON="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"
CTRF_JSON="${BENCHFLOW_REWARD_DETAILS_JSON:-${BENCHFLOW_CTRF_JSON:-/logs/verifier/ctrf.json}}"
PYTEST_BIN="${BENCHFLOW_PYTEST_BIN:-pytest}"

mkdir -p "$(dirname "$REWARD_TEXT")" "$(dirname "$REWARD_JSON")" "$(dirname "$CTRF_JSON")"

if command -v "$PYTEST_BIN" > /dev/null 2>&1; then
  # pytest is in the image (or named via BENCHFLOW_PYTEST_BIN) — use it.
  "$PYTEST_BIN" --ctrf "$CTRF_JSON" "$VERIFIER_DIR/test_outputs.py" -rA -v
else
  apt-get update
  apt-get install -y curl

  curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
  source $HOME/.local/bin/env

  uvx \
    --with pytest==8.4.1 \
    --with pytest-json-ctrf==0.3.5 \
    pytest --ctrf "$CTRF_JSON" "$VERIFIER_DIR/test_outputs.py" -rA -v
fi

PYTEST_EXIT_CODE=$?

if [ $PYTEST_EXIT_CODE -eq 0 ]; then
  REWARD=1.0
else
  REWARD=0.0
fi

printf '%s\n' "$REWARD" > "$REWARD_TEXT"
python3 - "$REWARD" "$REWARD_JSON" <<'PY'
import json
import sys

reward = float(sys.argv[1])
reward_json = sys.argv[2]
with open(reward_json, "w") as f:
    json.dump({"reward": reward}, f)
PY

exit 0
