# OpenCode GRPO rollout contract

PostTrain Arena uses TRL 1.8 `GRPOTrainer.rollout_func` to keep OpenCode as the
only agent harness during policy optimization.

## Runtime topology

The trainer reaches the TRL vLLM server directly. OpenCode reaches a small
OpenAI-compatible bridge in front of that same server:

- `TRL_VLLM_SERVER_BASE_URL`: trainer-side control URL used by TRL to
  synchronize policy weights;
- `BENCHFLOW_PROVIDER_BASE_URL`: public URL of `posttrainarena-train
  model-bridge`, reachable from the OpenCode process inside the BenchFlow
  sandbox.

`BENCHFLOW_ADAPTER_MODEL` is the model alias exposed by that server.
`BENCHFLOW_PROVIDER_API_KEY` carries its bearer credential when required.

The operator starts the synchronized server with `trl vllm-serve`, starts the
bridge with:

```bash
posttrainarena-train model-bridge \
  --upstream-url http://127.0.0.1:8000 \
  --tokenizer Qwen/Qwen3-4B \
  --tokenizer-revision <immutable-sha> \
  --port 8001
```

Only the bridge needs public ingress. It forwards chat messages and tool schemas
to TRL's `/chat/` endpoint, decodes the returned token IDs, parses Qwen
`<tool_call>` blocks into OpenAI tool calls, and emits OpenAI-compatible
streaming responses with sampled-token logprobs.

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

The executable pipeline pins BenchFlow
`93e58a2bd730a8ff3ca5aff5247aec845a370d1c`, which includes both sampled-token
logprob capture and the native TRL SFT converter.

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
