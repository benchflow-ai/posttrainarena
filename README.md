# PostTrain Arena

<!-- markdownlint-disable MD013 -->

[![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/mZ9Rc8q8W3) [![Pipeline CI](https://github.com/benchflow-ai/posttrainarena/actions/workflows/benchflow-posttrain-pipeline.yml/badge.svg)](https://github.com/benchflow-ai/posttrainarena/actions/workflows/benchflow-posttrain-pipeline.yml) [![License](https://img.shields.io/badge/License-AGPL--3.0-3C5440)](LICENSE)

**Contribute environments. Measure what models learn from them.** PostTrain Arena is an open project for agentic post-training and a proposed NeurIPS 2026 competition. Teams contribute task corpora; the organizer recipe trains a model on each corpus and measures transfer to held-out tasks.

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

## What teams submit

The competition unit is a **team corpus on one track**, not an individual task. Draft entry sizes are:

| Track | Submission | Minimum / recommended / maximum |
| --- | --- | --- |
| **Environment Submission** — headline track | Task packages containing `task.md`, `environment/`, `verifier/`, and `oracle/` | 50 / 100 / 200 |
| **Skill Learning** | `SKILL.md` packages | 20 / 50 / 100 |

Teams may enter both tracks as separate entries. Current checks warn below the minimum and fail above the maximum. See the [manifest contract](submissions/README.md) and [contribution guide](CONTRIBUTING.md).

For the environment track, the proposed workflow is:

```text
team environment corpus → fixed SFT + GRPO recipe → team checkpoint
                        → held-out evaluation → delta over reference
```

The draft scoring design uses a sealed 100-task evaluation suite and paired bootstrap confidence intervals; a 20-task public sample is planned for sanity checks. Skill entries use a frozen reference agent. Competition-scale execution and sealed evaluation are not established by the public canaries.

## Evidence and limitations

The public implementation includes task authoring tools and a BenchFlow + OpenCode + TRL training pipeline. The checked-in organizer recipe targets Qwen3.5-9B with Qwen3.5-397B-A17B teacher rollouts, one-epoch LoRA SFT, and LoRA GRPO. Recipes and historical results are separate from a guarantee that a new run will succeed.

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| [Local task tools](starting-kit/README.md) | Structural checks and Docker oracle/empty-trial replay | Complete schema validation, difficulty, leakage, or resistance to reward hacking |
| [Native-dataset OpenEnv smoke](docs/native-dataset-openenv-smoke.md) | Earlier one-train/one-eval pipeline executed end to end; score `0.0 → 0.0` | Model improvement or validation of the newer OpenCode path |
| [Qwen3.5 OpenCode canary](docs/qwen35-data-agent-e2e-canary.md) | 16 training / 14 disjoint same-domain eval tasks; SFT and 128 GRPO rollouts; `8/14 → 11/14` | Broad generalization; the diagnostic slice was not pre-registered and its paired 95% interval includes zero |
| [HF Jobs implementation and historical validation](docs/hf-jobs-validation.md) | Job bundles, secret boundaries, inspection, and publishing interfaces | Scheduler validation of every recipe or access to organizer resources |

The full public reference configuration selects 2,238 training tasks and 366 evaluation tasks. Having a configuration is not evidence that the full run completed. See [architecture and implementation status](docs/architecture-status.md) for compatibility details and the [documentation map](docs/README.md) for individual evidence reports.

### Hugging Face Submission Lab

The website's [training cookbook](https://posttrain.com/docs/cookbook) describes a separate, access-controlled demonstration. The lab, its Jobs namespace, and its artifacts require explicit access; cloning this repository does not grant that access. Its current reservation guard blocks repeat GPU submissions, and the normal submit form does not launch the separate checkpoint-search experiment.

A CPU planner success, optimizer training, saved-adapter reload, seen-task verifier success, and held-out evaluation are distinct milestones. In particular, verifier-guided supervised checkpoint search on a deliberately seen task is not GRPO or evidence of held-out generalization. For the public code path, start with [the training guide](docs/training-pipeline.md) and [HF Jobs operator guide](docs/hf-jobs.md).

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
