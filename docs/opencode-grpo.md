# OpenCode GRPO rollout contract

PostTrain Arena uses TRL 1.8 `GRPOTrainer.rollout_func` to keep OpenCode as the
only agent harness during policy optimization.

## Runtime topology

The trainer and OpenCode reach the same vLLM server through two addresses:

- `TRL_VLLM_SERVER_BASE_URL`: trainer-side control URL used by TRL to
  synchronize policy weights;
- `BENCHFLOW_PROVIDER_BASE_URL`: public OpenAI-compatible URL reachable from
  the OpenCode process inside the BenchFlow sandbox.

`BENCHFLOW_ADAPTER_MODEL` is the model alias exposed by that server.
`BENCHFLOW_PROVIDER_API_KEY` carries its bearer credential when required.

The operator starts the server with `trl vllm-serve` and exposes the inference
URL to Daytona or Docker. The local control URL and public inference URL must
route to the same server.

## Per-update lifecycle

```text
TRL GRPOTrainer
    -> synchronize current policy weights to vLLM
    -> call custom rollout_func
    -> run OpenCode through `bench eval run`
    -> execute task in BenchFlow sandbox
    -> run BenchFlow verifier
    -> read llm_trajectory.jsonl + reward
    -> return prompt IDs, completion IDs, sampled logprobs,
       environment mask, and verifier reward
    -> GRPO policy update
```

The BenchFlow LiteLLM proxy is invoked with
`BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`. It requests sampled-token logprobs from
the chat-completions endpoint and preserves them in
`trajectory/llm_trajectory.jsonl`.

The minimum pinned BenchFlow revision for this contract is
`c441b2abc07f48c03fd6638c5b9bcf7d837b6f38`.

The rollout parser reconstructs one causal sequence across all model turns.
Model-generated tokens receive action mask `1`; tool results, environment
feedback, and the next assistant-generation prefix receive mask `0`. Provider
token bytes are retokenized with the training tokenizer and must match the
provider token count. Any history drift, missing logprobs, malformed
trajectory, unscored rollout, agent/verifier error, zero-tool rollout, or token
budget overflow fails closed and is retried.

## Endpoint synchronization

- After SFT, the saved merged checkpoint is reloaded and synchronized to vLLM
  before post-SFT evaluation, including on resumed runs.
- Before each GRPO rollout batch, TRL synchronizes the current policy.
- After the final optimizer step, the saved GRPO checkpoint is reloaded and
  synchronized before held-out evaluation, including on resumed runs.

This removes the former TRL `environment_factory` agent loop. OpenEnv remains a
standalone protocol compatibility service, but it is not part of teacher
collection, evaluation, or GRPO rollout generation.

## Validation boundary

The token reconstruction, action masking, reward forwarding, retry policy,
vLLM wiring, and final synchronization are covered by no-spend tests. A live
GPU smoke additionally requires a reachable TRL vLLM server whose public
inference URL is accessible from the selected BenchFlow sandbox.
