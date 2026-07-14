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
CUDA_VISIBLE_DEVICES=1 posttrainarena-vllm-serve \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000

CUDA_VISIBLE_DEVICES=0 posttrainarena-train run \
  --config configs/qwen3.5-9b-data-agent-canary.toml
```

The wrapper rejects non-loopback `--host` values because TRL's generation and
weight-control endpoints are unauthenticated. Remote access must go through an
authenticated tunnel or proxy.

Using one GPU for both roles is unsupported by TRL's weight communicator. The
operator starts the bridge separately with:

```bash
posttrainarena-train model-bridge \
  --upstream-url http://127.0.0.1:8000 \
  --tokenizer Qwen/Qwen3.5-9B \
  --tokenizer-revision <immutable-sha> \
  --max-tokens 4096 \
  --max-context-tokens 49152 \
  --max-logprob-context-tokens 16384 \
  --max-sidecar-entries 2048 \
  --port 8001
```

Only the bridge needs public ingress. It forwards chat messages and tool schemas
to TRL's `/chat/` endpoint, decodes the returned token IDs, parses Qwen
`<tool_call>` blocks into OpenAI tool calls, including both JSON payloads and
Qwen3.5's native `<function=...><parameter=...>` syntax, and emits
OpenAI-compatible streaming responses. The 4,096-token per-call cap keeps the
non-streaming TRL server response below OpenCode's idle window; the GRPO
trajectory can still span multiple calls up to the recipe's aggregate
completion-token limit. Tool-bearing requests without explicit logprob capture
default to `temperature=1.0` with `seed=0`, preserving Qwen3.5's tool-use
behavior while making baseline/SFT/final agent decoding reproducible. OpenCode
helper requests without tools are not seeded, and GRPO logprob-capture requests
use the same temperature without a forced seed unless the caller explicitly
overrides it. Because LiteLLM's stream aggregation
does not retain choice-level logprobs, the bridge also keeps a bounded
authenticated sidecar keyed by the OpenAI completion ID. The GRPO collector
resolves the exact sampled token IDs/logprobs from that sidecar and writes
`grpo_tokens.json` beside each rollout attempt. The bridge caps each model turn
at 4,096 generated tokens while the pipeline separately enforces the
rollout-level completion budget.

On follow-up turns, OpenCode sends function arguments as JSON strings while
Qwen3.5's chat template expects mappings. The bridge normalizes those arguments
before forwarding the conversation to TRL; GRPO prompt reconstruction uses the
same normalization so served and trained token IDs stay aligned.
Before forwarding, the bridge tokenizes the complete Qwen prompt against the
configured 49,152-token context window. If tool output would overflow the
prompt budget, it preserves system/user messages and truncates the oldest tool
outputs with an explicit marker. Non-tool context overflow fails before calling
the TRL server.
Sampled-logprob GRPO requests use a stricter 16,384-token context cap so TRL can
recompute policy logprobs on one H100 without materializing 49k-token logits.
Ordinary baseline, SFT, and final evaluation keep the full context window.
The CLI also defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
reduce fragmentation across repeated variable-length GRPO batches.

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
For those explicit requests, the model bridge also keeps a one-shot sidecar
containing the exact prompt IDs, completion IDs, and sampled logprobs returned
by the TRL server. Ordinary evaluation requests do not create sidecars.

The executable pipeline pins BenchFlow
`cbc295464e62aa39f84e0daa675aa939c0e72f00`, which includes sampled-token
logprob capture, the native TRL SFT converter, and Qwen3.5 generation-prefix
validation.

The rollout parser reconstructs one causal sequence across all model turns
using the exact served prompt IDs from the bridge sidecar rather than
independently retokenizing the OpenAI request.
Model-generated tokens receive action mask `1`; tool results, environment
feedback, and the next assistant-generation prefix receive mask `0`. Provider
token bytes are retokenized only as a fallback and must match both the exact
served completion IDs and provider token count. BenchFlow call-purpose metadata excludes OpenCode helper
calls, and explicit failed provider attempts are ignored when OpenCode later
records a successful retry. If structured tool messages canonicalize a suffix
of prior sampled text, the parser rolls back only that suffix and masks the
canonical replacement as environment context. If OpenCode refreshes dynamic
system context or compacts history, reconstruction starts a new exact causal
segment at that request. Missing logprobs, malformed trajectories, unscored
rollouts, agent/verifier errors, or token-budget overflow fail closed and are
retried. Scored zero-tool completions remain valid negative rollouts.

## Endpoint synchronization

- Before baseline evaluation, the pinned base checkpoint is synchronized to
  vLLM so a reused server cannot retain weights from an earlier run.
- After SFT, the saved merged checkpoint is reloaded and synchronized to vLLM
  before post-SFT evaluation, including on resumed runs.
- Before each GRPO rollout batch, TRL synchronizes the current policy.
- After the final optimizer step, the saved GRPO checkpoint is reloaded and
  synchronized before held-out evaluation, including on resumed runs.

Every explicit base, SFT, and final synchronization is followed by a
deterministic direct-vs-public probe. The pipeline requires the TRL control
server, public OpenCode bridge, and private bridge sidecar to return identical
prompt IDs, completion IDs, and sampled logprobs before evaluation continues.

`posttrainarena-vllm-serve` uses TRL's server API and worker lifecycle, with one
Qwen3.5-specific compatibility mapping: Transformers trains the text policy as
`model.*`/`lm_head.*`, while vLLM serves the official multimodal checkpoint under
`language_model.*`. The worker prefixes only those synchronized text-policy
weights and leaves the frozen visual weights unchanged.

This removes the former TRL `environment_factory` agent loop. OpenEnv remains a
standalone protocol compatibility service, but it is not part of teacher
collection, evaluation, or GRPO rollout generation.
The bridge short-circuits OpenCode's title-generator prompt to a fixed local
title without a provider call, avoiding helper failures before the first tool
action.

## Validation boundary

The exact-ID reconstruction, action masking, reward forwarding, retry policy,
policy attestation, vLLM wiring, and final synchronization are covered by
no-spend tests. A post-run audit of the July 14 Qwen3.5 canary found that the
older implementation independently retokenized prompts: `0/321` sampled agent
exchanges matched the exact prompt-token count reported by the serving
endpoint. That historical run remains orchestration evidence, not valid GRPO
optimization evidence. The corrected exact-ID path requires a clean live
rerun.
