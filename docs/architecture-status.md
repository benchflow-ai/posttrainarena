# Architecture and implementation status

This document separates the PostTrain Arena vision from what is implemented in
the public repository today. It is the source of truth for compatibility and
roadmap claims.

## Vision

PostTrain Arena evaluates the quality of contributed agent environments by
holding the model recipe and held-out evaluation suite fixed:

```text
team task corpus
    -> organizer validation and verifier audit
    -> fixed SFT and optional RL recipe
    -> trained team checkpoint
    -> sealed held-out evaluation
    -> lift over a fixed reference checkpoint
```

The competition proposal currently describes a Qwen3-8B organizer recipe and a
private BenchFlow Signals evaluation suite. Those are draft competition rules,
not frozen implementation details.

## Current public implementation

The supported executable reference is the Qwen3-4B pipeline under
[`pipelines/benchflow-task-posttrain/`](../pipelines/benchflow-task-posttrain):

```text
PostTrain task lists and pinned HF snapshots
    -> direct BenchFlow or OpenEnv protocol integration
    -> BenchFlow task loading and sandbox lifecycle
    -> OpenCode teacher rollouts through BenchFlow
    -> verifier-approved, training-ready teacher trajectories
    -> TRL LoRA SFT and merged checkpoint
    -> OpenCode training-task reward gate through BenchFlow
    -> legacy TRL environment_factory optional GRPO
    -> OpenCode held-out evaluation and paired lift report
```

Teacher collection and every evaluation stage now use OpenCode as the agent
harness with required provider telemetry and BenchFlow artifact-health gates.
Only GRPO rollout generation still uses the older TRL-owned `run_bash` /
`submit` loop. Evaluation resolves the base and student model aliases plus the
OpenAI-compatible endpoint from named environment variables. The endpoint must
already expose the checkpoint selected for that stage; the GRPO migration owns
automatic policy-to-endpoint resynchronization.

The final machine-readable contract is:

```text
runs/<run-name>/reports/score.json
```

The checked-in smoke validates this orchestration and its zero-reward skip
path. It measured `0.0 -> 0.0`, so it is not evidence of model-quality lift or
competition-scale readiness.

## Ownership boundaries

| Layer | Current owner | Responsibility |
|---|---|---|
| Submission format | PostTrain Arena | `task.md`, `environment/`, `verifier/`, `oracle/`, and team manifests |
| Local author validation | PostTrain Arena | Structural checks, Docker oracle replay, and empty-trial rejection |
| Task/runtime/eval system | BenchFlow | Task snapshots, Daytona/Docker lifecycle, tools, verifier execution, rewards, and artifacts |
| Optimization | TRL | LoRA SFT and GRPO |
| Tracking | W&B | Training loss and GPU utilization when enabled |
| Task/model storage | Hugging Face Hub | Pinned snapshots and model/artifact publication when configured |

## Compatibility matrix

| Surface | Status | Evidence or boundary |
|---|---|---|
| PostTrain `task.md` packages | Implemented | Starting-kit examples, structural CI, local Docker harness, and the public 2,238-train/366-eval data-agent datasets |
| BenchFlow task-list training/eval | Implemented | Public pipeline, tests, CLI dry-run, and completed H100 smoke |
| Docker runtime | Implemented | Local author harness and BenchFlow runtime option |
| Daytona runtime | Implemented in pipeline | BenchFlow runtime option; credentials required for real execution |
| TRL SFT | Implemented | Tool-aware LoRA SFT and merged checkpoint path |
| TRL GRPO | Implemented | Reward-gated by default; explicit `always` policy supports zero-reward plumbing runs |
| OpenCode teacher collection | Implemented | Provider-qualified teacher model, required usage tracking, adaptive retries, and one training-ready rollout selected per task |
| OpenCode evaluation | Implemented | Baseline, post-SFT, training gate, final, and multi-benchmark evaluation all use `bench eval run --agent opencode`; the real SkillsBench + Daytona canary passed with complete telemetry and healthy trajectories |
| OpenCode GRPO | In migration | GRPO rollout generation and policy-to-endpoint resynchronization still use the legacy TRL environment loop |
| Harbor | Not a dependency | No Harbor adapter or trajectory translation is used |
| OpenEnv client/server lifecycle | Implemented | Pinned dependency, served adapter, typed client, real lifecycle tests, finalization, state, and session isolation |
| OpenEnv/BenchFlow Docker parity | Manually validated | Checked-in security task produced identical output and reward `1.0` through both integrations; CI uses a no-spend fake BenchFlow boundary |
| Native dataset OpenEnv pipeline | Prior end-to-end smoke validated | One train and one held-out native `task.md` package completed the earlier OpenEnv/TRL eval path; rerun the current OpenCode-eval path after a student endpoint is configured |
| Submission-to-recipe bridge | Implemented | Environment entries become pinned Hub datasets and portable recipes |
| HF Jobs execution | Implemented; scheduler credit blocked | UV job bundle and exact H100 runner validated; HF API allocation currently returns HTTP 402 until Jobs credits are granted |
| Hub artifact publishing | Implemented | Run reports, checkpoint provenance, logs, and failures publish to Hub datasets/models |
| Continuous leaderboard | Implemented | Atomic dataset records plus a deployable Gradio Space |
| Multi-benchmark evaluation | Implemented | One base/final checkpoint pair is evaluated across pinned suites with macro delta |
| Final Qwen3-8B competition recipe | Draft | Current reproducible reference pins Qwen3-4B |
| Demonstrated model-quality lift | Not yet | Reproduced smoke measured zero lift |

The OpenCode evaluation evidence is recorded in
[`opencode-evaluation-canary.md`](opencode-evaluation-canary.md).

## OpenEnv integration

OpenEnv is currently confined to the temporary legacy GRPO integration.
Teacher collection and evaluation invoke BenchFlow directly through the
OpenCode-backed `bench eval run` path.

The supported GRPO path is:

```text
TRL
    -> OpenEnv environment_factory adapter
    -> OpenEnv client/server protocol
    -> BenchFlow task, verifier, reward, and artifact engine
    -> Docker or Daytona
```

The implementation is intentionally a protocol layer:

```text
benchflow_pipeline/openenv/
  models.py
  client.py
  server.py
  tool_env.py
```

Real OpenEnv HTTP/WebSocket tests verify:

1. Start an OpenEnv server around the BenchFlow adapter.
2. Connect with the real OpenEnv client.
3. Call `reset`, execute one or more `step` actions, and retrieve state.
4. Submit or finalize an unsubmitted episode.
5. Return reward, completion state, and artifact references through OpenEnv
   models.
6. Reset again and prove sandbox isolation.
7. Keep concurrent sessions isolated.

A manual Docker parity canary also runs the same checked-in task and verifier
through the direct and OpenEnv integrations. Both paths produced identical
oracle output, reward `1.0`, and complete BenchFlow artifact trees. The
protocol lifecycle tests run in CI; the Docker canary remains an operator test.

Set `runtime.openenv_url` only for a separately deployed copy that shares the
pipeline's pinned task snapshots and BenchFlow artifact filesystem. When
omitted, the pipeline starts a local server. A general remote deployment still
requires task-resolution and artifact-transfer contracts. In both current
cases, BenchFlow remains the task/runtime/eval engine.

`posttrainarena-train openenv-serve` exposes the adapter as a discoverable
server command and resolves task IDs against server-owned pinned snapshots.
The managed HF Job uses co-located mode so task and artifact paths stay inside
one job. A general third-party artifact-transfer protocol remains separate.

The related upstream discussion is
[huggingface/OpenEnv#898](https://github.com/huggingface/OpenEnv/issues/898).
That issue proposes validation support inside OpenEnv. This repository's
compatibility comes from the adapter and lifecycle tests above.

Issue `#898` is therefore not a blocker for the competition runtime. It covers
authoring-time, spec-agnostic validation in the upstream OpenEnv CLI.

## Documentation precedence

Use this order when documents appear to differ:

1. Current code and tests under `pipelines/benchflow-task-posttrain/`
2. This architecture/status document
3. [`training-pipeline.md`](training-pipeline.md)
4. Draft competition language in the root README and contribution guide

Any PR that changes a compatibility status, runtime, model recipe, output
schema, or evidence claim must update this document in the same change.
