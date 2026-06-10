#!/bin/bash
# Reference solution for dogfood-hello-text. Writes the exact JSON the
# verifier expects.
set -e

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
mkdir -p "$WORKSPACE"

cat > "$WORKSPACE/answer.json" << 'JSON'
{
  "greeting": "hello",
  "subject": "post-training",
  "year": 2026
}
JSON

echo "Oracle wrote $WORKSPACE/answer.json"
