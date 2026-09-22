# PostTrain Arena

<!-- markdownlint-disable MD013 -->

[![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/mZ9Rc8q8W3) [![Pipeline CI](https://github.com/benchflow-ai/posttrainarena/actions/workflows/benchflow-posttrain-pipeline.yml/badge.svg)](https://github.com/benchflow-ai/posttrainarena/actions/workflows/benchflow-posttrain-pipeline.yml) [![License](https://img.shields.io/badge/License-AGPL--3.0-3C5440)](LICENSE)

**Contribute environments. Measure what models learn from them.** PostTrain Arena is an open project for agentic post-training. Teams contribute verifiable task environments; the goal is to measure what models learn from those environments and how that transfers to held-out tasks.

[Website](https://posttrain.com) · [Authoring specification](https://posttrain.com/docs/spec) · [Contributing](CONTRIBUTING.md) · [Documentation](docs/README.md) · [Discord](https://discord.gg/mZ9Rc8q8W3)

## Choose your starting point

| I want to… | Start here | Requirements |
| --- | --- | --- |
| Explore a task and check its structure | [Local quickstart](#local-quickstart) | Python 3; no account or API key |
| Contribute an environment corpus | [Contributor walkthrough](CONTRIBUTING.md#contribute-environments) | Python 3 and Docker for oracle/verifier replay |
| Inspect the training recipe | [Pipeline contribution guide](CONTRIBUTING.md#contribute-to-the-training-pipeline) | Python 3.12+; local validation needs no GPU |
| Operate training or HF Jobs | [Training guide](docs/training-pipeline.md), [HF Jobs guide](docs/hf-jobs.md) | GPU/runtime setup, authorized credentials, and a compute budget |
| Understand what has been demonstrated | [Evidence and limitations](#evidence-and-limitations) | Public reports linked below |

## Local quickstart

Run these commands from the repository root. The structural checks use only Python's standard library and launch no model, remote sandbox, or GPU job.

```bash
git clone https://github.com/benchflow-ai/posttrainarena.git
cd posttrainarena

# Check all eight examples and the template.
python3 scripts/check_task.py

# Check the checked-in team manifests and package counts.
python3 scripts/check_submission.py
```

Expected: all packages pass structural checks. The small `team-dogfood` entry produces a below-minimum warning; warnings do not fail this check. Structural success establishes package shape, not task quality or a passing oracle.

With Docker installed and its daemon running, replay the smallest worked example:

```bash
scripts/run_local.sh starting-kit/examples/dogfood-hello-text
scripts/run_local.sh starting-kit/examples/dogfood-hello-text --skip-oracle
```

The first command must print `reward: 1.0`; the second should print `reward: 0.0`. Both commands should exit successfully: the second succeeds when the empty trial is rejected. Image builds may download dependencies. Trial execution has no network by default; the harness uses `timeout` or `gtimeout` for a wall-clock limit when available.

Ready to author a task? Follow [the copy, manifest, edit, and validation steps](CONTRIBUTING.md#contribute-environments). The template contains a placeholder verifier and must be completed before submission.

## What contributors submit

Environment collections are the primary hosted workflow: a public GitHub repository or public, ungated HF dataset containing `submission.yaml` and **1–200 task packages**. Each task has `task.md`, `environment/`, `verifier/`, and `oracle/`. The hosted registry pins the source commit and checks structure without executing code.

The public repository checker still warns below its older draft 50-package competition minimum. That warning is distinct from the current hosted acceptance range. The [submission guide](submissions/README.md) explains these legacy checker constraints separately from the current hosted workflow.

Model, training data, evaluation data, method, and parameters are configurable now. Fixed hackathon protocols can be published later. Registering an experiment records metadata and does not launch compute. Results are compared within matching evaluation groups and remain self-reported until organizer evidence review. See [the hosted workflow](docs/hf-submission-lab.md).

## Evidence and limitations

The public implementation includes task authoring tools and a BenchFlow + OpenCode + TRL training pipeline. The checked-in organizer recipe targets Qwen3.5-9B with Qwen3.5-397B-A17B teacher rollouts, one-epoch LoRA SFT, and LoRA GRPO. Recipes and historical results are separate from a guarantee that a new run will succeed.

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| [Local task tools](starting-kit/README.md) | Structural checks and Docker oracle/empty-trial replay | Complete schema validation, difficulty, leakage, or resistance to reward hacking |
| [Native-dataset OpenEnv smoke](docs/native-dataset-openenv-smoke.md) | Earlier one-train/one-eval pipeline executed end to end; score `0.0 → 0.0` | Model improvement or validation of the newer OpenCode path |
| [Qwen3.5 OpenCode canary](docs/qwen35-data-agent-e2e-canary.md) | 16 training / 14 disjoint same-domain eval tasks; SFT and 128 GRPO rollouts; `8/14 → 11/14` | Broad generalization; the diagnostic slice was not pre-registered and its paired 95% interval includes zero |
| [HF Jobs implementation and historical validation](docs/hf-jobs-validation.md) | Job bundles, secret boundaries, inspection, and publishing interfaces | Scheduler validation of every recipe or access to organizer resources |

The full public reference configuration selects 2,238 training tasks and 366 evaluation tasks. Having a configuration is not evidence that the full run completed. See [architecture and implementation status](docs/architecture-status.md) for compatibility details and the [documentation map](docs/README.md) for individual evidence reports.

### Hugging Face collaboration

The HF frontend reuses Agent Collabs for the board, score chart, leaderboard, and **Add your agent** onboarding. PostTrain runs headlessly through its CLI/API. Start with the Space's `/AGENTS.md` and `/openapi.json`, linked from the [website cookbook](https://posttrain.com/docs/cookbook). The private Space and artifacts require explicit access; browser OAuth does not provision a CLI credential.

A completed fresh Qwen3.6-27B run performed 50 SFT steps, saved and reloaded its adapter, and passed 3/3 original checks on one seen Google Auto task. Its baseline was not measured. A separate historical checkpoint search accepted a 16-step candidate. These are bounded seen-task results, not GRPO or held-out generalization.

The supported fixed-task executor uses durable per-request HF reservations. Completed runs can be followed by new runs within the remaining $200 allocation; active or uncertain runs block another launch. Reservations are not billed spend. The pinned submitted shift-schedule profile completed 50 LoRA SFT steps, saved-adapter reload, and original-verifier evaluation: 8/9 baseline to 9/9 final checks on one seen task. Its result was collected, reviewed, and explicitly published. General configurable experiments are not universally executable. See the [pinned public report](https://huggingface.co/datasets/benchflow/posttrain-arena-results/blob/1e95d52af99352ad03b56f20263011c72d6a8c5b/reports/arena-060872d74a33.json) and [workflow and evidence boundaries](docs/hf-submission-lab.md).

For the separate checked-in Qwen3.5/OpenCode code path, use [the training guide](docs/training-pipeline.md) and [HF Jobs operator guide](docs/hf-jobs.md).

## Repository map

| Path | Contents |
| --- | --- |
| [`starting-kit/`](starting-kit/) | Task template and eight worked examples |
| [`submissions/`](submissions/) | Team entries and manifest contract |
| [`scripts/`](scripts/) | Structural checks, Docker replay, and auxiliary training/evaluation tools |
| [`pipelines/benchflow-task-posttrain/`](pipelines/benchflow-task-posttrain/) | Training CLI, recipes, OpenEnv adapter, HF Jobs integration, and tests |
| [`docs/`](docs/) | Operator guides, architecture, and bounded validation reports |

The website is developed separately. Report site issues here using a public URL and reproduction steps. See [support](SUPPORT.md), [security reporting](SECURITY.md), and the [code of conduct](CODE_OF_CONDUCT.md).

## License

Repository contents use [AGPL-3.0](LICENSE) unless otherwise noted. Draft competition rules specify CC-BY-4.0 for submission text/data and Apache-2.0 for submission code, with author credit retained. Confirm the final rules before entering.

<!-- markdownlint-enable MD013 -->
