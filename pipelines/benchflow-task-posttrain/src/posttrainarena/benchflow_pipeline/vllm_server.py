"""TRL vLLM server with Qwen3.5 text-to-multimodal weight-name mapping."""

from __future__ import annotations

import json
import ipaddress
import os
import signal
import sys
from collections.abc import Sequence
from multiprocessing.connection import Connection
from typing import Any

from trl.scripts.vllm_serve import WeightSyncWorkerExtension


def server_weight_name(model: Any, name: str) -> str:
    """Map Transformers text-model names onto vLLM's Qwen3.5 wrapper."""
    if (
        hasattr(model, "language_model")
        and not name.startswith("language_model.")
        and name.startswith(("model.", "lm_head."))
    ):
        return f"language_model.{name}"
    return name


def validate_control_host(host: str) -> None:
    """Keep the unauthenticated TRL control API on loopback."""
    if host == "localhost":
        return
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError(
            "posttrainarena-vllm-serve must bind to a loopback host; "
            "use an authenticated tunnel or proxy for remote access"
        )


class PostTrainWeightSyncWorkerExtension(WeightSyncWorkerExtension):
    """Accept text-only trainer weights for multimodal Qwen3.5 serving."""

    def update_named_param(
        self,
        name: str,
        dtype: str,
        shape: Sequence[int],
    ) -> None:
        mapped_name = server_weight_name(self.model_runner.model, name)
        super().update_named_param(mapped_name, dtype, shape)


def llm_worker(
    script_args: Any,
    data_parallel_rank: int,
    master_port: int,
    connection: Connection,
) -> None:
    """Launch one TRL vLLM worker with the PostTrain weight-sync extension."""
    from vllm import LLM

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    llm = None
    try:
        llm = LLM(
            model=script_args.model,
            revision=script_args.revision,
            tensor_parallel_size=script_args.tensor_parallel_size,
            gpu_memory_utilization=script_args.gpu_memory_utilization,
            enforce_eager=script_args.enforce_eager,
            dtype=script_args.dtype,
            enable_prefix_caching=script_args.enable_prefix_caching,
            kv_cache_dtype=script_args.kv_cache_dtype,
            max_model_len=script_args.max_model_len,
            worker_extension_cls=(
                "posttrainarena.benchflow_pipeline.vllm_server."
                "PostTrainWeightSyncWorkerExtension"
            ),
            trust_remote_code=script_args.trust_remote_code,
            model_impl=script_args.vllm_model_impl,
            distributed_executor_backend=script_args.distributed_executor_backend,
            logprobs_mode="processed_logprobs",
            speculative_config=(
                json.loads(script_args.speculative_config)
                if script_args.speculative_config
                else None
            ),
        )
        connection.send({"status": "ready"})

        while True:
            try:
                command = connection.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if command["type"] in {"call", "fire_and_forget"}:
                method = getattr(llm, command["method"])
                result = method(
                    *command.get("args", ()),
                    **command.get("kwargs", {}),
                )
                if command["type"] == "call":
                    connection.send(result)
            elif command["type"] == "shutdown":
                break
    finally:
        if llm is not None:
            try:
                llm.collective_rpc(method="close_communicator")
            except Exception:
                pass
        connection.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: list[str] | None = None) -> int:
    """Run TRL's server API with the PostTrain worker extension."""
    from trl.scripts import vllm_serve

    parser = vllm_serve.make_parser(prog="posttrainarena-vllm-serve")
    (script_args,) = parser.parse_args_and_config(
        args=sys.argv[1:] if argv is None else argv
    )
    validate_control_host(script_args.host)
    vllm_serve.llm_worker = llm_worker
    vllm_serve.main(script_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
