#!/usr/bin/env bash
set +e
V="${BENCHFLOW_VERIFIER_DIR:-/verifier}"; T="${BENCHFLOW_REWARD_TEXT:-/logs/verifier/reward.txt}"; J="${BENCHFLOW_REWARD_JSON:-/logs/verifier/reward.json}"; C="${BENCHFLOW_REWARD_DETAILS_JSON:-/logs/verifier/ctrf.json}"; mkdir -p "$(dirname "$T")"; pytest --ctrf "$C" "$V/test_outputs.py" -q; c=$?; [ "$c" -eq 0 ] && r=1.0 || r=0.0; echo "$r" > "$T"; python3 - "$r" "$J" <<'PY'
import json,sys
with open(sys.argv[2],"w") as f:json.dump({"reward":float(sys.argv[1])},f)
PY
exit 0
