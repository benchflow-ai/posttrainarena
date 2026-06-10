#!/bin/bash
# Reference solution. Runs inside the same container the agent uses.
# Must achieve a passing score so reviewers can confirm the task is
# solvable. CI re-runs this on every image bump.
set -e

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
mkdir -p "$WORKSPACE"
export BENCHFLOW_WORKSPACE="$WORKSPACE"

# Example: write a trivial passing answer. Replace with your real solution.
cat > "$WORKSPACE/answer.json" << 'JSON'
{
  "result": 42
}
JSON

echo "Oracle wrote $WORKSPACE/answer.json"
