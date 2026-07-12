# BenchFlow Task-List Post-Training Pipeline

This is the public, reproducible training path for running SFT and optional
GRPO over BenchFlow-compatible task suites. OpenCode drives teacher collection,
evaluation, and GRPO rollouts. TRL owns optimization and synchronizes the
current policy to the shared vLLM endpoint.

The repository-wide architecture and compatibility status is documented in
[`docs/architecture-status.md`](../../docs/architecture-status.md). In
particular, this package provides OpenEnv protocol compatibility and the HF
Jobs/Hub publishing path.

The interface is intentionally small:

```text
training task list + held-out eval task list + TOML recipe
    -> baseline eval
    -> verifier-approved teacher trajectories
    -> tool-aware SFT
    -> post-SFT reward gate
    -> optional GRPO
    -> held-out score and paired lift report
```

BenchFlow owns task snapshots, sandbox lifecycle, verifiers, rollout artifacts,
and paired evaluation. OpenCode owns teacher, evaluation, and GRPO model
interaction. TRL owns SFT and GRPO optimization. OpenEnv remains a standalone
compatibility service, not a second training runtime or eval engine. The default
recipes pin the public BenchFlow-native `task.md` datasets
`benchflow/data_agent_rl_environment_train` and
`benchflow/data_agent_rl_environment_eval`. The pipeline does not depend on
Harbor or translate Harbor trajectories.
## Repository Layout

```text
benchflow-task-posttrain/
  configs/                 checked-in, reviewable training recipes
  task-lists/              explicit train and eval task IDs
  scripts/bootstrap_gpu.sh GPU-host bootstrap
  src/.../config.py        typed TOML contract and validation
  src/.../pipeline.py      resumable stage orchestration
  src/.../teacher.py       verified OpenCode teacher rollouts
  src/.../opencode.py      OpenCode baseline/gate/final evaluation
  src/.../grpo.py          OpenCode custom rollouts for TRL GRPO
  src/.../sft.py           tool-aware LoRA SFT and weight merge
  src/.../openenv/         OpenEnv client/server protocol adapter
  tests/                   no-spend contract tests
```

Generated runs are written under `runs/` and ignored by Git.

The same package also prepares team submissions, submits HF UV Jobs, publishes
Hub artifacts, evaluates benchmark matrices, serves OpenEnv, and deploys the
leaderboard. See [`docs/hf-jobs.md`](../../docs/hf-jobs.md).

## Quick Start

Python 3.12 and `uv` are required.

```bash
cd pipelines/benchflow-task-posttrain
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[train,test]'
source .venv/bin/activate
```

Validate and inspect the recipe without credentials or GPU spend:

```bash
posttrainarena-train validate \
  --config configs/qwen3-4b-data-agent-smoke.toml

posttrainarena-train plan \
  --config configs/qwen3-4b-data-agent-smoke.toml \
  --run-name local-review

posttrainarena-train run \
  --config configs/qwen3-4b-data-agent-smoke.toml \
  --run-name local-review \
  --dry-run
```

For a GPU host, `scripts/bootstrap_gpu.sh` installs this package and its pinned
BenchFlow dependency into an isolated virtual environment. The script pins the
CUDA 12.8 Torch wheel and fails immediately if the GPU is unavailable.

Use `configs/qwen3-4b-data-agent-forced-grpo-smoke.toml` to force the GRPO
plumbing path. It sets `grpo.run_policy = "always"` so a
zero-reward GRPO run can validate plumbing; production recipes should normally
retain `run_policy = "on_reward"`.

## Credentials

Load credentials from a secret manager or an untracked environment file. Do
not place them in TOML recipes, task lists, command-line arguments, or commits.

The example recipe requires:

- `HF_TOKEN` for task snapshots and optional artifact publication
- `DAYTONA_API_KEY` for `runtime.sandbox = "daytona"`
- `GLM_API_KEY` and `GLM_BASE_URL` for the example OpenCode teacher model
- `BENCHFLOW_BASE_MODEL` and `BENCHFLOW_ADAPTER_MODEL` for the served base and
  current-student model aliases
- `BENCHFLOW_PROVIDER_BASE_URL` and `BENCHFLOW_PROVIDER_API_KEY` for the
  OpenAI-compatible endpoint used by OpenCode evaluation
- `TRL_VLLM_SERVER_BASE_URL` for TRL's weight-sync/control connection to the
  same student server
- `WANDB_API_KEY` when `tracking.report_to = "wandb"`
- any verifier-specific credentials required by the selected task packages

Provider credential values are never written to the run plan or score report.

## Run

```bash
posttrainarena-train run \
  --config configs/qwen3-4b-data-agent-smoke.toml \
  --run-name qwen3-4b-data-agent
```

Use `--resume` after interruption. Completed snapshots, evaluations, and
checkpoints are reused when their expected marker artifacts exist.

The public OpenCode endpoint and `TRL_VLLM_SERVER_BASE_URL` must route to the
same vLLM server. The pipeline synchronizes SFT weights before SFT evaluation,
the current GRPO policy before each rollout batch, and final weights before the
held-out evaluation.

The final contract is:

```text
runs/<run-name>/reports/score.json
```

Important fields include `baseline_score`, `sft_score`, `grpo_gate_score`,
`score_after_posttrain`, `delta_score`, `grpo_planned`, `grpo_ran`, exact task
IDs, dataset revisions, BenchFlow commit, `grpo_run_policy`, and the recorded
stage commands. A dry-run may set `grpo_planned` while leaving `grpo_ran` false.

## Reward Gate

GRPO runs only when the post-SFT score on the configured training-task gate is
at least `grpo.threshold`. This prevents spending GPU time on a constant-zero
reward distribution. A skipped GRPO stage is a valid pipeline result, not a
runtime failure.

`grpo.run_policy = "always"` bypasses the reward decision for end-to-end
plumbing validation. It is not a recommendation for useful RL training.

Do not use held-out eval tasks to tune this gate. Production recipes should use
separate training, gate/development, and final evaluation lists.

## Reproduced Smoke Result

The checked-in recipe mirrors the completed public smoke:

- 15 training tasks produced 15 verifier-approved teacher trajectories
- 40 LoRA SFT steps completed and merged into standalone Qwen3-4B weights
- baseline score: `0.0` on two held-out tasks
- SFT score: `0.0`
- four-task GRPO gate: `0.0`
- GRPO was correctly skipped
- final paired delta: `0.0`

This validates the end-to-end system and reward gate. It is not evidence of a
model-quality lift; meaningful claims require larger training and held-out sets.

## Contributing

Contributions should preserve the task-list-in, score-out contract and keep
provider, runtime, and trainer logic behind their current module boundaries.

Before opening a PR:

```bash
python3 -m pytest pipelines/benchflow-task-posttrain/tests -q
python3 -m py_compile \
  pipelines/benchflow-task-posttrain/src/posttrainarena/benchflow_pipeline/*.py \
  pipelines/benchflow-task-posttrain/src/posttrainarena/benchflow_pipeline/openenv/*.py
```

New recipes should pin dataset revisions and model revisions, use new task-list
files, document expected compute, and default to no-spend tests. Never commit
checkpoints, trajectories, raw provider responses, or credentials.

The CI protocol tests use OpenEnv's real HTTP/WebSocket transport with a fake
BenchFlow boundary. Before changing runtime semantics, also run a real Docker
parity canary against one checked-in task and compare reward plus artifact
trees across `integration = "benchflow"` and `integration = "openenv"`.
