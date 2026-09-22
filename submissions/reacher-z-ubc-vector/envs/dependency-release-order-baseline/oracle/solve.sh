#!/usr/bin/env bash
set -euo pipefail
python3 "${BENCHFLOW_ORACLE_DIR:-/oracle}/solve.py"
