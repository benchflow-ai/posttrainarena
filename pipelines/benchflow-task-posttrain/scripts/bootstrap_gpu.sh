#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/benchflow-ai/posttrainarena.git}"
REPO_REF="${REPO_REF:-main}"
WORK_ROOT="${WORK_ROOT:-$HOME/posttrainarena}"

sudo apt-get update
sudo apt-get install -y git git-lfs jq
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -d "$WORK_ROOT/.git" ]]; then
  git clone "$REPO_URL" "$WORK_ROOT"
fi
git -C "$WORK_ROOT" fetch origin "$REPO_REF"
git -C "$WORK_ROOT" checkout --detach FETCH_HEAD

uv venv "$WORK_ROOT/.venv" --python 3.12
uv pip install --python "$WORK_ROOT/.venv/bin/python" \
  -e "$WORK_ROOT/pipelines/benchflow-task-posttrain[train]"

cat >"$WORK_ROOT/activate-posttrain.sh" <<EOF
export PATH="$WORK_ROOT/.venv/bin:\$PATH"
cd "$WORK_ROOT/pipelines/benchflow-task-posttrain"
EOF

echo "Bootstrap complete. Source $WORK_ROOT/activate-posttrain.sh."
