from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_bootstrap_selects_cuda_matched_vllm_wheel() -> None:
    script = (ROOT / "scripts" / "bootstrap_gpu.sh").read_text()

    assert 'VLLM_VERSION="${VLLM_VERSION:-0.23.0}"' in script
    assert "git git-lfs jq ninja-build" in script
    assert 'shutil.which("ninja")' in script
    assert 'export VLLM_CUDA_VARIANT="${VLLM_CUDA_VARIANT:-cu129}"' in script
    assert 'VLLM_WHEEL_BUILD="${VLLM_WHEEL_BUILD:-0fc695' in script
    assert 'VLLM_WHEEL_ARCH="x86_64"' in script
    assert 'VLLM_WHEEL_ARCH="aarch64"' in script
    assert 'VLLM_WHEEL_URL="https://wheels.vllm.ai/${VLLM_WHEEL_BUILD}/' in script
    assert '--index-url "$PYTORCH_WHEEL_INDEX"' in script
    assert '"$VLLM_WHEEL_URL"' in script
    assert "unsafe-best-match" not in script
    assert "expected_variant = f\"+{os.environ['VLLM_CUDA_VARIANT']}\"" in script
    assert "PYTORCH_CUDA_ALLOC_CONF" in script
    assert "expandable_segments:True" in script
