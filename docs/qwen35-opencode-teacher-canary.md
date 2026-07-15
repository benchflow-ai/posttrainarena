# Qwen3.5 OpenCode teacher canary

On July 13, 2026, the Qwen3.5-397B-A17B teacher completed a real OpenCode
rollout through BenchFlow on the checked-in `dogfood-hello-text` task.

## Pinned inputs

- Declared teacher source: `Qwen/Qwen3.5-397B-A17B`
- Declared source revision: `8472618112abcbd45acbcdc58436aff4233c23f7`
- Runtime route: `openrouter/qwen/qwen3.5-397b-a17b`
- BenchFlow: `cbc295464e62aa39f84e0daa675aa939c0e72f00`
- Agent harness: OpenCode
- Sandbox: Docker

## Result

- Reward: `1.0`
- Agent or verifier errors: `0`
- Tool calls: `2`
- Provider exchanges: `3`
- Token usage: `22,614`
- Telemetry coverage: `100%`
- `results.jsonl`: `training_ready=true`
- `llm_trajectory.jsonl`: present and structurally valid
- `acp_trajectory.jsonl`: present and structurally valid

The rollout used OpenCode's own write and read tools, then completed normally.
It did not use another agent loop.
OpenRouter returned the exact model ID, but does not expose a Hugging Face
commit fingerprint; the source revision is provenance metadata, not a
cryptographically enforced deployment identity.

## Qwen3.5 TRL conversion

The rollout exposed a Qwen3.5 chat-template edge case: adding the assistant
completion rewrites the final generation-prefix token, so the old strict
prompt-prefix check rejected an otherwise valid row. BenchFlow PR `#929`,
merged as `cbc295464e62aa39f84e0daa675aa939c0e72f00`, changes the validation
boundary to allow drift only after the assistant generation mask begins.

With that fix applied, the real rollout produced and validated:

- TRL SFT rows: `3`
- Rows with tool calls: `3`
- Maximum tokenized row: `7,465`
- Minimum trainable assistant tokens: `26`
- Context-compacted rows: `0`

## Claim boundary

This historical canary proves the Qwen3.5 teacher, OpenCode tool loop,
BenchFlow telemetry, raw trajectory capture, and Qwen3.5-aware TRL conversion
on one real task. The later native Linux GPU canary validates Qwen3.5-9B
SFT/GRPO and exploratory same-domain uplift on a 16-train/14-eval slice.
Neither run is the full 2,238-task public-reference execution or a
participant-corpus/private-eval competition run. See
[`qwen35-data-agent-e2e-canary.md`](qwen35-data-agent-e2e-canary.md).
