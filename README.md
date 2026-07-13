# PostTrain Arena

<!-- markdownlint-disable MD013 -->

[![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/mZ9Rc8q8W3) [![GitHub](https://img.shields.io/github/stars/benchflow-ai/posttrainarena?style=social)](https://github.com/benchflow-ai/posttrainarena) [![Website](https://img.shields.io/badge/Website-posttrain.com-3C5440)](https://posttrain.com) [![Pipeline CI](https://github.com/benchflow-ai/posttrainarena/actions/workflows/benchflow-posttrain-pipeline.yml/badge.svg)](https://github.com/benchflow-ai/posttrainarena/actions/workflows/benchflow-posttrain-pipeline.yml) [![License](https://img.shields.io/badge/License-AGPL--3.0-3C5440)](LICENSE)

The open arena for post-training: contribute agentic RL environments, then measure what the resulting model generalizes to unseen tasks.

**[Website](https://posttrain.com)** · **[Authoring spec](https://posttrain.com/docs/spec)** · **[Training pipeline](./docs/training-pipeline.md)** · **[Architecture status](./docs/architecture-status.md)** · **[Contributing](./CONTRIBUTING.md)** · **[Discord](https://discord.gg/mZ9Rc8q8W3)**

> [!IMPORTANT]
> PostTrain Arena is a proposed NeurIPS 2026 competition. The checked-in organizer recipe now targets `Qwen/Qwen3.5-9B`, uses `Qwen/Qwen3.5-397B-A17B` teacher rollouts, and runs one-epoch LoRA SFT followed by LoRA GRPO. The older Qwen3-4B path has end-to-end execution evidence; the Qwen3.5 recipe has not yet demonstrated model-quality lift.

## How it works

Most competitions fix the environment and ask teams to submit an agent. PostTrain Arena inverts that contract: teams contribute environment corpora, and the organizers hold the post-training recipe and evaluation suite fixed.

```text
team task corpus
    → fixed SFT + GRPO recipe
    → trained team checkpoint
    → sealed held-out evaluation
    → Δ over a fixed reference checkpoint
```

The headline track rewards environments that teach capabilities which transfer beyond their own training tasks—not environments that only improve in-domain performance.

## Competition at a glance

| Track | What a team submits | Per-entry scale | Evaluation |
| --- | --- | --- | --- |
| **Track 2 — Environment Submission** (headline) | Containerized task packages: `task.md` + `environment/` + `verifier/` + `oracle/` | 50 minimum / 100 recommended / 200 maximum | Managed SFT→GRPO, then held-out generalization delta |
| **Track 1 — Skill Learning** | Modular `SKILL.md` packages | 20 minimum / 50 recommended / 100 maximum | Pass@1 of a frozen reference agent |

- **Scoring.** Track 2 uses `Δ = pass rate(team checkpoint) − pass rate(reference checkpoint)` on a sealed 100-task suite, with paired bootstrap confidence intervals. A 20-task public sample is reserved for sanity checks.
- **Phases.** Phase 0 is a public-sample warm-up, Phase 1 provides development feedback, and Phase 2 freezes submissions for private evaluation. Teams may enter both tracks as separate entries.
- **Open release.** Under the draft rules, accepted environments, teacher data, and trained checkpoints are released openly while authors retain credit.

## Public implementation status

The checked-in implementation now includes the full public-data organizer recipe plus a 1x1 canary; live Qwen3.5 validation and the sealed competition eval remain pending.

| Surface | Current public status |
| --- | --- |
| Participant task format and local validation | **Implemented** — eight worked examples, structural checks, Docker oracle replay, and empty-trial rejection |
| BenchFlow task-list training and evaluation | **Implemented** — pinned snapshots, one verified teacher rollout per training task, one-epoch LoRA SFT, LoRA GRPO over the training set, held-out evaluation, and score reports |
| OpenCode agent harness | **Implemented end to end** — teacher collection, baseline/gate/final eval, benchmark matrices, and TRL custom GRPO rollouts use OpenCode; TRL synchronizes the current policy to the shared vLLM endpoint |
| Public data | **Available** — [2,238 training tasks](https://huggingface.co/datasets/benchflow/data_agent_rl_environment_train) and [366 held-out evaluation tasks](https://huggingface.co/datasets/benchflow/data_agent_rl_environment_eval) in native `task.md` format |
| OpenEnv protocol path | **Implemented** — served adapter, typed client, lifecycle tests, Docker parity validation, and a native-dataset end-to-end smoke |
| HF Jobs execution | **Implemented** — portable UV job bundles, pinned code refs, named-secret boundaries, status inspection, and Hub publishing; live scheduler allocation currently awaits HF credits |
| Continuous leaderboard | **Implemented** — atomic Hub dataset records and a deployable Gradio Space |
| Multi-benchmark evaluation | **Implemented** — one base/final checkpoint pair can be scored across pinned Data Agent and SkillsBench suites |
| Qwen3.5-9B organizer recipe | **Implemented, validation pending** — full 2,238-train/366-eval config is checked in; real quality lift is not yet proven |
| Demonstrated model-quality lift | **Not yet** — completed smokes validated system mechanics, not learning gains |

> [!NOTE]
> On July 10, 2026, a real one-train/one-held-out run completed snapshotting, baseline evaluation, verifier-approved teacher collection, LoRA SFT, a forced GRPO step, final evaluation, and artifact publication through the earlier OpenEnv/TRL evaluation path. Scores remained `0.0 → 0.0`, so this is evidence of end-to-end operability—not quality improvement and not validation of the newer OpenCode evaluation path. See the [native-dataset OpenEnv smoke report](https://github.com/benchflow-ai/posttrainarena/blob/main/docs/native-dataset-openenv-smoke.md).

For compatibility details and evidence boundaries, use [Architecture and implementation status](./docs/architecture-status.md) as the source of truth.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`starting-kit/`](./starting-kit) | Task template and organizer-authored examples |
| [`submissions/`](./submissions) | Team entries and `submission.yaml` contract |
| [`scripts/`](./scripts) | Self-contained structural checks and local Docker harness |
| [`pipelines/benchflow-task-posttrain/`](./pipelines/benchflow-task-posttrain) | Public BenchFlow + OpenEnv + TRL training implementation |
| [`docs/`](./docs) | Architecture, operator guide, and validation evidence |

The examples under `starting-kit/` are reference material, not competition entries.

## Quick start

Local task authoring requires Python 3 and Docker. It does not require BenchFlow, a GPU, or provider API keys.

```bash
git clone https://github.com/benchflow-ai/posttrainarena.git
cd posttrainarena

mkdir -p submissions/your-team/envs
cp -R starting-kit/template submissions/your-team/envs/your-env-name

# Edit the task package and add submissions/your-team/submission.yaml.
python3 scripts/check_task.py submissions/your-team/envs
python3 scripts/check_submission.py

# The oracle must score 1.0.
scripts/run_local.sh submissions/your-team/envs/your-env-name

# A do-nothing trial must fail.
scripts/run_local.sh submissions/your-team/envs/your-env-name --skip-oracle
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the submission workflow, validation ladder, and reviewer checklist. Organizers and researchers should start with the [training pipeline guide](./docs/training-pipeline.md) and [HF Jobs handoff](./docs/hf-jobs.md).

## Documentation

- [Task authoring specification](https://posttrain.com/docs/spec)
- [Architecture and implementation status](./docs/architecture-status.md)
- [Training pipeline operator guide](./docs/training-pipeline.md)
- [OpenCode GRPO rollout contract](./docs/opencode-grpo.md)
- [OpenCode SFT-to-GRPO smoke](./docs/opencode-grpo-smoke.md)
- [Qwen3.5 OpenCode teacher canary](./docs/qwen35-opencode-teacher-canary.md)
- [OpenCode evaluation canary](./docs/opencode-evaluation-canary.md)
- [Hugging Face Jobs and leaderboard handoff](./docs/hf-jobs.md)
- [HF handoff validation report](./docs/hf-jobs-validation.md)
- [Native-dataset OpenEnv smoke report](https://github.com/benchflow-ai/posttrainarena/blob/main/docs/native-dataset-openenv-smoke.md)
- [Starting-kit guide](./starting-kit/README.md)
- [Team submission guide](./submissions/README.md)
- [Security](./SECURITY.md) · [Support](./SUPPORT.md) · [Code of conduct](./CODE_OF_CONDUCT.md)

Questions are welcome on [Discord](https://discord.gg/mZ9Rc8q8W3) or by email at [labs@benchflow.ai](mailto:labs@benchflow.ai).

## License

Repository contents are licensed under [AGPL-3.0](./LICENSE) unless noted otherwise. Under the draft competition rules, submissions use CC-BY-4.0 for text and data and Apache-2.0 for code.

<!-- markdownlint-enable MD013 -->
