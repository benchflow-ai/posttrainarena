# BenchFlow Task-List Post-Training Pipeline

This is the public, reproducible training path for running SFT and GRPO over
BenchFlow-compatible task suites. OpenCode drives teacher collection,
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
    -> native TRL prompt/completion/tools SFT
    -> post-SFT reward gate
    -> LoRA GRPO over the training set
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
  src/.../grpo.py          OpenCode custom rollouts for TRL LoRA GRPO
  src/.../sft.py           TRL completion-only/assistant-only LoRA SFT
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

Validate and inspect the Qwen3.5 canary without credentials or GPU spend:

```bash
posttrainarena-train validate \
  --config configs/qwen3.5-9b-data-agent-canary.toml

posttrainarena-train plan \
  --config configs/qwen3.5-9b-data-agent-canary.toml \
  --run-name local-review

posttrainarena-train run \
  --config configs/qwen3.5-9b-data-agent-canary.toml \
  --run-name local-review \
  --dry-run
```

`configs/qwen3.5-9b-data-agent-full.toml` is the organizer recipe. It selects
all 2,238 public training tasks and all 366 public held-out tasks, requires one
verified trajectory per training task, performs one SFT epoch, and always runs
one GRPO epoch. The canary uses the same models and harness on a 1x1 task split.

For a GPU host, `scripts/bootstrap_gpu.sh` installs this package and its pinned
BenchFlow dependency into an isolated virtual environment. The script resolves
Torch 2.11 and vLLM 0.20+ with `UV_TORCH_BACKEND=auto`, the compatible runtime
line for Qwen3.5 plus Transformers 5.6+, and fails immediately if the GPU is
unavailable.

## Credentials

Load credentials from a secret manager or an untracked environment file. Do
not place them in TOML recipes, task lists, command-line arguments, or commits.

The example recipe requires:

- `HF_TOKEN` for task snapshots and optional artifact publication
- `DAYTONA_API_KEY` for `runtime.sandbox = "daytona"`
- `OPENROUTER_API_KEY` for the Qwen3.5-397B-A17B OpenCode teacher
- `BENCHFLOW_BASE_MODEL` and `BENCHFLOW_ADAPTER_MODEL` for the served base and
  current-student model aliases
- `BENCHFLOW_PROVIDER_BASE_URL` and `BENCHFLOW_PROVIDER_API_KEY` for the
  OpenAI-compatible endpoint used by OpenCode evaluation
- `BENCHFLOW_MODEL_BRIDGE_CONTROL_URL` for trainer-local logprob retrieval when
  the public provider URL uses separate ingress
- `TRL_VLLM_SERVER_BASE_URL` for TRL's weight-sync/control connection to the
  same student server
- `WANDB_API_KEY` when `tracking.report_to = "wandb"`
- any verifier-specific credentials required by the selected task packages

The trainer and TRL vLLM server must use different physical CUDA devices. On a
two-GPU host, use `CUDA_VISIBLE_DEVICES=1` for `trl vllm-serve` and
`CUDA_VISIBLE_DEVICES=0` for `posttrainarena-train run`.

Provider credential values are never written to the run plan or score report.

## Run

```bash
posttrainarena-train run \
  --config configs/qwen3.5-9b-data-agent-full.toml \
  --run-name qwen35-9b-data-agent
```

Use `--resume` after interruption. Completed snapshots, evaluations, and
checkpoints are reused when their expected marker artifacts exist.

The public OpenCode endpoint is `posttrainarena-train model-bridge`, which
forwards to the TRL server at `TRL_VLLM_SERVER_BASE_URL`. The pipeline
synchronizes SFT weights before SFT evaluation, the current GRPO policy before
each rollout batch, and final weights before the held-out evaluation.

The final contract is:

```text
runs/<run-name>/reports/score.json
```

Important fields include `baseline_score`, `sft_score`, `grpo_gate_score`,
`score_after_posttrain`, `delta_score`, `grpo_planned`, `grpo_ran`, exact task
IDs, dataset revisions, BenchFlow commit, `grpo_run_policy`, and the recorded
stage commands. The report also records the SFT and GRPO adapter and merged
checkpoint paths. A dry-run may set `grpo_planned` while leaving `grpo_ran`
false.

## Reward Gate

The Qwen3.5 full recipe sets `grpo.run_policy = "always"` because GRPO is part
of the fixed organizer recipe. The engine still supports `on_reward` for
low-cost experiments that should skip a constant-zero reward distribution.

Do not use held-out eval tasks to tune this gate. Production recipes should use
separate training, gate/development, and final evaluation lists.

## Historical Qwen3-4B Smoke Result

The retained Qwen3-4B smoke recipe mirrors the completed public smoke:

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
