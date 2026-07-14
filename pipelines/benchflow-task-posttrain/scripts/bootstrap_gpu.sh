#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/benchflow-ai/posttrainarena.git}"
REPO_REF="${REPO_REF:-main}"
WORK_ROOT="${WORK_ROOT:-$HOME/posttrainarena}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
export VLLM_CUDA_VARIANT="${VLLM_CUDA_VARIANT:-cu129}"
VLLM_WHEEL_BUILD="${VLLM_WHEEL_BUILD:-0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665}"
PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/${VLLM_CUDA_VARIANT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$(uname -m)" in
  x86_64) VLLM_WHEEL_ARCH="x86_64" ;;
  aarch64 | arm64) VLLM_WHEEL_ARCH="aarch64" ;;
  *)
    echo "Unsupported vLLM wheel architecture: $(uname -m)" >&2
    exit 1
    ;;
esac
VLLM_WHEEL_URL="https://wheels.vllm.ai/${VLLM_WHEEL_BUILD}/vllm-${VLLM_VERSION}%2B${VLLM_CUDA_VARIANT}-cp38-abi3-manylinux_2_28_${VLLM_WHEEL_ARCH}.whl"

sudo apt-get update
sudo apt-get install -y git git-lfs jq ninja-build
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -d "$WORK_ROOT/.git" ]]; then
  git clone "$REPO_URL" "$WORK_ROOT"
fi
git -C "$WORK_ROOT" fetch origin "$REPO_REF"
git -C "$WORK_ROOT" checkout --detach FETCH_HEAD

uv venv "$WORK_ROOT/.venv" --python 3.12
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-$VLLM_CUDA_VARIANT}"
uv pip install --python "$WORK_ROOT/.venv/bin/python" \
  --torch-backend "$UV_TORCH_BACKEND" \
  --index-url "$PYTORCH_WHEEL_INDEX" \
  "torch==2.11.0"
uv pip install --python "$WORK_ROOT/.venv/bin/python" \
  "$VLLM_WHEEL_URL"
uv pip install --python "$WORK_ROOT/.venv/bin/python" \
  -e "$WORK_ROOT/pipelines/benchflow-task-posttrain[train]"

"$WORK_ROOT/.venv/bin/python" - <<'PY'
import importlib.metadata
import os
import shutil

import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit(f"CUDA unavailable with torch {torch.__version__}")
if shutil.which("ninja") is None:
    raise SystemExit("ninja is required for FlashInfer kernel JIT compilation")
vllm_version = importlib.metadata.version("vllm")
expected_variant = f"+{os.environ['VLLM_CUDA_VARIANT']}"
if not vllm_version.endswith(expected_variant):
    raise SystemExit(
        f"vLLM wheel mismatch: expected {expected_variant}, got {vllm_version}"
    )
print(
    "CUDA ready: "
    f"torch {torch.__version__}, vLLM {vllm_version}, "
    f"{torch.cuda.get_device_name(0)}"
)
PY

cat >"$WORK_ROOT/activate-posttrain.sh" <<EOF
export PATH="$WORK_ROOT/.venv/bin:\$PATH"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$WORK_ROOT/pipelines/benchflow-task-posttrain"
EOF

echo "Bootstrap complete. Source $WORK_ROOT/activate-posttrain.sh."
