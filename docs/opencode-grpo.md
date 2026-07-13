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
  sandbox;
- `BENCHFLOW_MODEL_BRIDGE_CONTROL_URL`: optional trainer-local URL for the
  authenticated logprob sidecar. It falls back to
  `BENCHFLOW_PROVIDER_BASE_URL` when both paths are the same.

`BENCHFLOW_ADAPTER_MODEL` is the model alias exposed by that server.
`BENCHFLOW_PROVIDER_API_KEY` carries its bearer credential when required.

TRL server-mode synchronization requires the trainer and vLLM worker to use
different physical CUDA devices. On a two-GPU host, a minimal split is:

```bash
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000

CUDA_VISIBLE_DEVICES=0 posttrainarena-train run \
  --config configs/qwen3.5-9b-data-agent-canary.toml
```

Using one GPU for both roles is unsupported by TRL's weight communicator. The
operator starts the bridge separately with:

```bash
posttrainarena-train model-bridge \
  --upstream-url http://127.0.0.1:8000 \
  --tokenizer Qwen/Qwen3.5-9B \
  --tokenizer-revision <immutable-sha> \
  --max-tokens 4096 \
  --port 8001
```

Only the bridge needs public ingress. It forwards chat messages and tool schemas
to TRL's `/chat/` endpoint, decodes the returned token IDs, parses Qwen
`<tool_call>` blocks into OpenAI tool calls, and emits OpenAI-compatible
streaming responses. Because LiteLLM's stream aggregation does not retain
choice-level logprobs, the bridge also keeps a bounded authenticated sidecar
keyed by the OpenAI completion ID. The GRPO collector resolves the exact sampled
token IDs/logprobs from that sidecar and writes `grpo_tokens.json` beside each
rollout attempt. The bridge caps each model turn at 4,096 generated tokens while
the pipeline separately enforces the rollout-level completion budget.

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
`cbc295464e62aa39f84e0daa675aa939c0e72f00`, which includes sampled-token
logprob capture, the native TRL SFT converter, and Qwen3.5 generation-prefix
validation.

The rollout parser reconstructs one causal sequence across all model turns.
Model-generated tokens receive action mask `1`; tool results, environment
feedback, and the next assistant-generation prefix receive mask `0`. Provider
token bytes are retokenized with the training tokenizer and must match the
provider token count. BenchFlow call-purpose metadata excludes OpenCode helper
calls, and explicit failed provider attempts are ignored when OpenCode later
records a successful retry. If structured tool messages canonicalize a suffix
of prior sampled text, the parser rolls back only that suffix and masks the
canonical replacement as environment context. If OpenCode refreshes dynamic
system context or compacts history, reconstruction starts a new exact causal
segment at that request. Missing logprobs, malformed trajectories, unscored
rollouts, agent/verifier errors, zero-tool rollouts, or token-budget overflow
fail closed and are retried.

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
vLLM wiring, and final synchronization are covered by no-spend tests. The real
two-H100 SkillsBench + Daytona validation is documented in
[`opencode-grpo-smoke.md`](opencode-grpo-smoke.md). That run proves execution
and synchronization but reported no reward or held-out score lift.
