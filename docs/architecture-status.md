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
| PostTrain `task.md` packages | Implemented | Starting-kit examples, structural CI, and local Docker harness |
| BenchFlow task-list training/eval | Implemented | Public pipeline, tests, CLI dry-run, and completed H100 smoke |
| Docker runtime | Implemented | Local author harness and BenchFlow runtime option |
| Daytona runtime | Implemented in pipeline | BenchFlow runtime option; credentials required for real execution |
| TRL SFT | Implemented | Tool-aware LoRA SFT and merged checkpoint path |
| TRL GRPO | Implemented and reward-gated | Runs only after a training-task reward gate passes |
| Harbor | Not a dependency | No Harbor adapter or trajectory translation is used |
| OpenEnv client/server lifecycle | **Not implemented** | No `openenv` dependency, `openenv.yaml`, `EnvClient`, served environment, or lifecycle test exists |
| HF Jobs execution | **Not implemented** | No HF Jobs launcher or deployment workflow exists |
| Final Qwen3-8B competition recipe | Draft | Current reproducible reference pins Qwen3-4B |
| Demonstrated model-quality lift | Not yet | Reproduced smoke measured zero lift |

## OpenEnv roadmap

The current repository is **not OpenEnv-compatible**. A BenchFlow environment
method named `reset` is not evidence of OpenEnv compatibility. Credible
compatibility requires an actual OpenEnv client/server surface and an
end-to-end lifecycle test.

The minimum adapter should provide:

```text
integrations/openenv/
  openenv.yaml
  models.py
  client.py
  server/
    app.py
    posttrain_environment.py
  tests/
    test_lifecycle.py
    test_task_adapter.py
```

Acceptance requires all of the following:

1. Start an OpenEnv-compatible server around a checked-in PostTrain task.
2. Connect with the real OpenEnv client.
3. Call `reset`, execute one or more `step` actions, and retrieve state.
4. Terminate or submit the episode and run the existing verifier.
5. Return reward, completion state, and artifact references through OpenEnv
   models.
6. Reset again and prove sandbox isolation.
7. Pass a no-secret Docker lifecycle test in CI.
8. Document the mapping between one-shot PostTrain trials and long-lived
   OpenEnv episodes.

The related upstream discussion is
[huggingface/OpenEnv#898](https://github.com/huggingface/OpenEnv/issues/898).
That issue proposes validation support inside OpenEnv; it does not make this
repository OpenEnv-compatible.

## Documentation precedence

Use this order when documents appear to differ:

1. Current code and tests under `pipelines/benchflow-task-posttrain/`
2. This architecture/status document
3. [`training-pipeline.md`](training-pipeline.md)
4. Draft competition language in the root README and contribution guide

Any PR that changes a compatibility status, runtime, model recipe, output
schema, or evidence claim must update this document in the same change.
