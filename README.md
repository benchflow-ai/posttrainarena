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
| Submit a collection to the hosted Arena and run it on a challenge | [Hosted guide](docs/hf-submission-lab.md) | A public repository or HF dataset, Python 3.10+, and a Hugging Face token |
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

Expected: all packages pass structural checks. Structural success establishes package shape, not task quality, metadata values, or a passing oracle.

With Docker installed and its daemon running, replay the smallest worked example:

```bash
scripts/run_local.sh starting-kit/examples/dogfood-hello-text
scripts/run_local.sh starting-kit/examples/dogfood-hello-text --skip-oracle
```

The first command must print `reward: 1.0`; the second should print `reward: 0.0`. Both commands should exit successfully: the second succeeds when the empty trial is rejected. Image builds may download dependencies. Trial execution has no network by default; the harness uses `timeout` or `gtimeout` for a wall-clock limit when available.

Ready to author a task? Follow [the copy, manifest, edit, and validation steps](CONTRIBUTING.md#contribute-environments). The template contains a placeholder verifier and must be completed before submission.

## What contributors submit

You contribute one thing: the training data. A challenge fixes the base model, the post-training recipe and a sealed held-out suite, and your collection of task environments is what the model trains on. A collection is a public GitHub repository or public, ungated HF dataset containing a flat `submission.yaml` and **1–200 task packages** under `envs/`. Each task has `task.md`, `environment/`, `verifier/`, and `oracle/`, and should declare its author, license, category and origin, which is how contributors are credited. The Arena pins the source commit and checks structure without executing code; the [submission guide](submissions/README.md) describes the layout.

Each run evaluates the base model on the challenge's held-out suite, trains it on your tasks with the recipe, and evaluates it again. The score is the change: held-out pass rate after training minus before, in percentage points, measured in the same run. A BenchFlow editor reviews each collected result, and the leaderboard ranks collections by their mean change over accepted runs. See [how to submit a collection and run it](docs/hf-submission-lab.md).

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

The Arena runs on a public Hugging Face Space, <https://benchflow-posttrain-arena.hf.space>. Its front page, `/`, is the Agent Collabs board, where participants and organizers discuss work; the submissions app at [`/arena`](https://benchflow-posttrain-arena.hf.space/arena) lists challenges, collections, tasks, runs and the leaderboard, and has a Submit a collection form. Reading needs no sign-in. Validating, submitting and preflighting need a Hugging Face identity (sign in, or give the CLI any valid HF token); launching and collecting a run are limited to the collection's author and BenchFlow editors. HF Jobs pages and the runs and artifact datasets are private to BenchFlow.

Agents use the headless CLI, `arena_cli.py`, which calls the same API as the app; the Space's [`/AGENTS.md`](https://benchflow-posttrain-arena.hf.space/AGENTS.md) is the reference. Validate and submit a collection, preflight a run on a challenge with `run --challenge`, follow it with `runs --run-id`, and collect the scored result with `result collect`. Every run draws on the Arena's one shared compute cap, which does not reset: `python3 arena_cli.py budget` shows what remains, and a challenge's health says whether it takes runs right now. Challenge runs execute this repository's pipeline at a commit each challenge pins; [the training guide](docs/training-pipeline.md) and [HF Jobs operator guide](docs/hf-jobs.md) cover running it yourself.

Results from before challenges, on the retired experiment path, are seen-task practice: LoRA SFT on Qwen3.6-27B, evaluated on the task it trained on. A fresh Google Auto run passed 3/3 original checks after 50 SFT steps (its baseline was not measured), and the submitted shift-schedule run went from 8/9 to 9/9 checks and was reviewed and published. Neither is GRPO or held-out generalization. The [hosted guide's history section](docs/hf-submission-lab.md#history-the-retired-experiment-path) keeps the details, and the [pinned public report](https://huggingface.co/datasets/benchflow/posttrain-arena-results/blob/1e95d52af99352ad03b56f20263011c72d6a8c5b/reports/arena-060872d74a33.json) records the shift-schedule result.

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
