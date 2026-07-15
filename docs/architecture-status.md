# Architecture and implementation status

<!-- markdownlint-disable MD013 MD060 -->

This document separates the PostTrain Arena vision from what is implemented in
the public repository today. It is the source of truth for compatibility and
roadmap claims.

## Vision

PostTrain Arena evaluates the quality of contributed agent environments by
holding the model recipe and held-out evaluation suite fixed:

```text
team task corpus
    -> organizer validation and verifier audit
    -> fixed SFT and GRPO recipe
    -> trained team checkpoint
    -> sealed held-out evaluation
    -> lift over a fixed reference checkpoint
```

The organizer implementation targets Qwen3.5-9B with a private BenchFlow
Signals evaluation suite. The exploratory public 16-train/14-eval same-domain
canary validates the recipe and model update path; the competition rules, final
compute budget, and private-suite scale remain draft.

## Current public implementation

The supported executable reference is the Qwen3.5-9B pipeline under
[`pipelines/benchflow-task-posttrain/`](../pipelines/benchflow-task-posttrain):

```text
PostTrain task lists and pinned HF snapshots
    -> BenchFlow task loading and sandbox lifecycle
    -> pinned Qwen3.5-9B synchronization and OpenCode baseline evaluation
    -> OpenCode teacher rollouts through BenchFlow
    -> one verifier-approved Qwen3.5-397B-A17B trajectory per training task
    -> one-epoch TRL LoRA SFT, adapter, and merged checkpoint
    -> OpenCode training-task reward gate through BenchFlow
    -> TRL GRPOTrainer custom rollout_func
    -> OpenCode rollouts through BenchFlow and current student endpoint
    -> token IDs, sampled logprobs, action mask, and verifier reward
    -> LoRA policy update, adapter/merged export, and vLLM resynchronization
    -> OpenCode held-out evaluation and paired lift report
```

Teacher collection and every evaluation stage now use OpenCode as the agent
harness with required provider telemetry and BenchFlow artifact-health gates.
GRPO uses TRL 1.8's custom rollout function, not `environment_factory`.
Evaluation resolves the base and student model aliases plus the
OpenAI-compatible endpoint from named environment variables. TRL synchronizes
the pinned base policy before baseline evaluation, the SFT policy before its
evaluation, the current GRPO policy before each rollout batch, and the final
policy before held-out evaluation.

The final machine-readable contract is:

```text
runs/<run-name>/reports/score.json
```

The full checked-in public reference recipe covers all 2,238 training tasks and
all 366 held-out evaluation tasks at immutable pinned Hub revisions.
Competition recipes keep the model and optimizer contract fixed while
replacing the training dataset/task list with one participant corpus and the
eval dataset/task list with the organizer's sealed internal tasks. The
exploratory 16-train/14-eval public canary completed the corrected path and
observed a same-domain pass-rate increase from `8/14` to `11/14`; the full
public-reference run and private competition readiness remain unproven.

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
| BenchFlow task-list training/eval | Implemented | Public pipeline, exact-ID and package-content isolation checks, content-addressed resume validation, CLI dry-run, and completed H100 orchestration smoke |
| Docker runtime | Implemented | Local author harness and BenchFlow runtime option |
| Daytona runtime | Implemented in pipeline | BenchFlow runtime option; credentials required for real execution |
| TRL SFT | Implemented | BenchFlow `trl-sft` prompt/completion/tools conversion, tokenizer-aware message windows, exact common-prefix token labels for Qwen3.5, and merged checkpoint path |
| TRL GRPO | Implemented | Qwen3.5 full recipe always runs one epoch over all training tasks; the custom OpenCode rollout function returns token IDs, sampled logprobs, action mask, and BenchFlow verifier reward; optimization is LoRA without quantization |
| OpenCode teacher collection | Implemented | Provider-qualified Qwen3.5-397B-A17B teacher, required usage tracking, adaptive retries, and fail-closed one-training-ready-rollout-per-task coverage |
| OpenCode evaluation | Implemented and live Qwen3.5 validated | Baseline, post-SFT, training gate, final, and multi-benchmark evaluation all use `bench eval run --agent opencode`; real SkillsBench and Qwen3.5 Data Agent canaries produced complete telemetry and healthy trajectories |
| OpenCode GRPO | Implemented and live Qwen3.5 validated | TRL custom rollout function invokes OpenCode/BenchFlow, consumes exact served prompt/completion IDs plus sampled logprobs, forwards verifier reward, rejects zero-variance/no-update adapters, and resynchronizes the vLLM endpoint; the corrected run completed 128 rollouts and updated all 248 LoRA-B tensors |
| Harbor | Not a dependency | No Harbor adapter or trajectory translation is used |
| OpenEnv client/server lifecycle | Implemented | Pinned dependency, served adapter, typed client, real lifecycle tests, finalization, state, and session isolation |
| OpenEnv/BenchFlow Docker parity | Manually validated | Checked-in security task produced identical output and reward `1.0` through both integrations; CI uses a no-spend fake BenchFlow boundary |
| Native Data Agent pipeline | Live canary validated | Sixteen training and 14 disjoint evaluation task IDs from the same source dataset completed strict teacher collection, LoRA SFT, 128 OpenCode GRPO rollouts, synchronization, and paired evaluation with an exploratory `8/14 -> 11/14` increase |
| Submission-to-recipe bridge | Implemented | Environment entries become pinned Hub datasets and portable recipes |
| HF Jobs execution | Canary handoff implemented; current allocation unverified | UV job bundle and historical H100 runner validated; July 11 scheduler requests were credit-blocked, and no paid scheduler launch was submitted during the July 15 documentation audit; the Docker-based Qwen3.5 full recipe currently targets a persistent native Linux GPU host |
| Hub artifact publishing | Implemented | Run reports, checkpoint provenance, logs, and failures publish to Hub datasets/models |
| Continuous leaderboard | Implemented | Atomic dataset records plus a deployable Gradio Space |
| Multi-benchmark evaluation | Implemented | One base/final checkpoint pair is evaluated across pinned suites with macro delta |
| Qwen3.5-9B public reference recipe | Implemented and canary validated | Immutable base/data revisions, declared Qwen3.5-397B-A17B teacher provenance, all-task teacher coverage, one-epoch LoRA SFT, one-epoch LoRA GRPO, exact served token IDs, effective-update gates, and endpoint attestation are checked in; full 2,238/366 and private competition execution remain pending |
| Observed canary uplift | Exploratory same-domain evidence only | Corrected run increased `8/14 -> 11/14` with zero regressions; the diagnostic slice was not a pre-registered generalization benchmark and the paired 95% interval includes zero |

The OpenCode evaluation evidence is recorded in
[`opencode-evaluation-canary.md`](opencode-evaluation-canary.md).
The GRPO rollout and endpoint contract is documented in
[`opencode-grpo.md`](opencode-grpo.md).
The real two-H100 SkillsBench + Daytona run is recorded in
[`opencode-grpo-smoke.md`](opencode-grpo-smoke.md).
The real Qwen3.5 Data Agent run is recorded in
[`qwen35-data-agent-e2e-canary.md`](qwen35-data-agent-e2e-canary.md).

## OpenEnv integration

OpenEnv is a standalone protocol compatibility surface. The current training
pipeline does not use it as an agent harness or GRPO transport; teacher
collection, evaluation, and GRPO all invoke BenchFlow through OpenCode.

The supported protocol path is:

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

`posttrainarena-train openenv-serve` exposes the adapter as a discoverable
server command and resolves task IDs against server-owned pinned snapshots.
A general third-party artifact-transfer protocol remains separate.

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
