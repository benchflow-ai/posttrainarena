# Hugging Face Jobs handoff

This is the runnable handoff from a PostTrain Arena team entry to Hugging Face
compute and the continuous leaderboard.

## Ownership

- PostTrain Arena validates submissions, emits pinned recipes, and orchestrates
  jobs and publishing.
- BenchFlow owns task loading, sandbox lifecycle, verifiers, rewards, and
  rollout artifacts.
- OpenEnv is a standalone compatibility service; the current training job uses
  OpenCode through BenchFlow for teacher, evaluation, and GRPO rollouts.
- TRL owns SFT and GRPO optimization.
- HF Jobs runs the pinned UV script; Hub datasets/models/Spaces store results.

Upstream OpenEnv issue
[`#898`](https://github.com/huggingface/OpenEnv/issues/898) proposes a generic
authoring validator. It is useful, but it is not required by this runtime path.

## Install

```bash
cd pipelines/benchflow-task-posttrain
python3.12 -m venv .venv
.venv/bin/pip install -e '.[train,hf,test]'
source .venv/bin/activate
```

Load provider credentials from an untracked environment file. The launcher
sends only explicitly named secrets to the HF Job API. Values are never written
into the bundle, plan, score, or leaderboard.

## 1. Prepare a team corpus

```bash
posttrainarena-train prepare-submission \
  --entry submissions/<team-entry> \
  --base-config configs/qwen3.5-9b-data-agent-full.toml \
  --out .local/prepared/<team-entry> \
  --dataset-repo <namespace>/posttrainarena-<team-entry> \
  --upload
```

The output contains the reviewed dataset staging tree, portable train/eval task
lists, immutable recipe, and a machine-readable preparation manifest. Uploaded
submission datasets are private by default; use `--public` only during the
explicit post-competition release.

## 2. Test the HF path without GPU spend

Inspect the portable bundle without creating any remote resource:

```bash
posttrainarena-train hf-job-submit \
  --config configs/qwen3.5-9b-data-agent-canary.toml \
  --bundle-dir .local/hf-jobs/local-plan \
  --run-id local-plan-001 \
  --submission-id organizer-smoke \
  --team-name Organizers \
  --artifact-repo <namespace>/posttrainarena-results \
  --leaderboard-repo <namespace>/posttrainarena-leaderboard \
  --flavor cpu-basic \
  --pipeline-dry-run \
  --launcher-dry-run
```

Then run the same bundle on a real CPU Job:

```bash
posttrainarena-train hf-job-submit \
  --config configs/qwen3.5-9b-data-agent-canary.toml \
  --bundle-dir .local/hf-jobs/cpu-smoke \
  --run-id cpu-smoke-001 \
  --submission-id organizer-smoke \
  --team-name Organizers \
  --artifact-repo <namespace>/posttrainarena-results \
  --leaderboard-repo <namespace>/posttrainarena-leaderboard \
  --posttrainarena-ref "$(git rev-parse HEAD)" \
  --flavor cpu-basic \
  --pipeline-dry-run \
  --wait
```

## 3. Run the Qwen3.5 SFT→GRPO canary and multi-benchmark eval

```bash
posttrainarena-train hf-job-submit \
  --config configs/qwen3.5-9b-data-agent-canary.toml \
  --benchmarks configs/multi-benchmark-smoke.toml \
  --bundle-dir .local/hf-jobs/h100-smoke \
  --run-id h100-smoke-001 \
  --submission-id organizer-smoke \
  --team-name Organizers \
  --artifact-repo <namespace>/posttrainarena-results \
  --model-repo <namespace>/posttrainarena-qwen35-9b \
  --leaderboard-repo <namespace>/posttrainarena-leaderboard \
  --posttrainarena-ref "$(git rev-parse HEAD)" \
  --namespace <namespace> \
  --flavor h100 \
  --timeout 6h \
  --private-artifacts \
  --wait
```

After the canary passes, replace the config with
`configs/qwen3.5-9b-data-agent-full.toml` for the organizer run. Keep
`--private-artifacts` for any run that uses the sealed internal evaluation set;
score reports contain exact task IDs and rollout evidence.
Model repositories are also private by default. `--public-model` is reserved
for the explicit release workflow after the competition.

Default full-run secrets are derived from the recipe's teacher provider. The
Qwen3.5 organizer recipe requires `HF_TOKEN`, `OPENROUTER_API_KEY`,
`BENCHFLOW_BASE_MODEL`,
`BENCHFLOW_ADAPTER_MODEL`, `BENCHFLOW_PROVIDER_BASE_URL`,
`BENCHFLOW_PROVIDER_API_KEY`, and `TRL_VLLM_SERVER_BASE_URL`. Override the list
with repeated `--secret-env NAME`.

The GPU job also needs a TRL-compatible vLLM server plus
`posttrainarena-train model-bridge`. Its trainer-side control URL is
`TRL_VLLM_SERVER_BASE_URL`; `BENCHFLOW_PROVIDER_BASE_URL` must expose the bridge
to OpenCode inside Daytona. Set `BENCHFLOW_MODEL_BRIDGE_CONTROL_URL` when the
trainer should retrieve sampled logprobs through a local bridge URL. The
checked-in UV runner does not create public ingress automatically, so the
operator must provision that endpoint before launching the current
OpenCode-GRPO path. TRL server mode also requires the trainer and vLLM worker to
run on different physical CUDA devices.

Caught failures publish the run's reports, rollout jobs, and checkpoints back
to the private artifact repository. Reusing the same run ID restores that state
and invokes `posttrainarena-train run --resume`; an interrupted GRPO stage is
restarted from the SFT checkpoint rather than reusing stale-policy rollouts.
Hard provider termination can still prevent the final failure upload, so the
competition-scale recipe should use a persistent GPU host until stage-level
checkpoint uploads are added.

The canonical Qwen3.5 recipe uses Docker. Run it on a Docker-enabled native
Linux GPU host; a generic managed Job container may not expose a Docker daemon.

## 4. Inspect a job

```bash
posttrainarena-train hf-job-status \
  --job-id <job-id> \
  --namespace <namespace>

hf jobs logs <job-id> --namespace <namespace>
```

## 5. Deploy the live leaderboard

```bash
posttrainarena-train deploy-leaderboard \
  --leaderboard-repo <namespace>/posttrainarena-leaderboard \
  --space-repo <namespace>/posttrainarena-leaderboard
```

Each run upserts job/submission identity, status, exact revisions, baseline,
final and delta scores, per-benchmark results, macro delta, and artifact/model
URLs. Updates use an optimistic parent commit and retry concurrent writers.

## OpenEnv server command

```bash
posttrainarena-train openenv-serve \
  --tasks-dir .local/data-agent-eval \
  --include-task 0000_369_369503_qa_1 \
  --environment daytona \
  --jobs-dir .local/openenv-jobs \
  --host 0.0.0.0 \
  --port 8000
```

The server resolves client task IDs to its own pinned task paths. Competition
HF Jobs normally use co-located mode so all artifacts remain directly
publishable by the job runner.

## Validation and current HF blocker

The exact UV runner completed the full submission-to-training-to-leaderboard
flow on an H100, including Data Agent and SkillsBench evaluation. See
[`hf-jobs-validation.md`](hf-jobs-validation.md).

HF Jobs allocation itself currently returns HTTP 402 for all available
namespaces because prepaid Jobs credits are unavailable. After HF enables the
grant or billing balance, rerun the documented `hf-job-submit` command without
any code change.
