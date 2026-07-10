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
    -> BenchFlow run_bash / submit agent harness
    -> verifier-approved teacher trajectories
    -> TRL LoRA SFT and merged checkpoint
    -> BenchFlow reward gate on training tasks
    -> optional TRL GRPO
    -> held-out BenchFlow evaluation and paired lift report
```

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
| Harbor | Not a dependency | No Harbor adapter or trajectory translation is used |
| OpenEnv client/server lifecycle | Implemented | Pinned dependency, served adapter, typed client, real lifecycle tests, finalization, state, and session isolation |
| OpenEnv/BenchFlow Docker parity | Manually validated | Checked-in security task produced identical output and reward `1.0` through both integrations; CI uses a no-spend fake BenchFlow boundary |
| HF Jobs execution | **Not implemented** | No HF Jobs launcher or deployment workflow exists |
| Final Qwen3-8B competition recipe | Draft | Current reproducible reference pins Qwen3-4B |
| Demonstrated model-quality lift | Not yet | Reproduced smoke measured zero lift |

## OpenEnv integration

The supported path is:

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

The related upstream discussion is
[huggingface/OpenEnv#898](https://github.com/huggingface/OpenEnv/issues/898).
That issue proposes validation support inside OpenEnv. This repository's
compatibility comes from the adapter and lifecycle tests above.

## Documentation precedence

Use this order when documents appear to differ:

1. Current code and tests under `pipelines/benchflow-task-posttrain/`
2. This architecture/status document
3. [`training-pipeline.md`](training-pipeline.md)
4. Draft competition language in the root README and contribution guide

Any PR that changes a compatibility status, runtime, model recipe, output
schema, or evidence claim must update this document in the same change.
