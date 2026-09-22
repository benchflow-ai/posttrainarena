# Documentation map

<!-- markdownlint-disable MD013 MD060 -->

## Start here

- **First contribution:** run the [local quickstart](../README.md#local-quickstart), then follow the [environment walkthrough](../CONTRIBUTING.md#contribute-environments).
- **Recipe exploration:** use the [local validate/plan/dry-run commands](../CONTRIBUTING.md#inspect-a-recipe-without-training) before installing a GPU runtime.
- **Training operations:** read [the pipeline guide](training-pipeline.md), then [HF Jobs](hf-jobs.md) if using remote compute.
- **Evidence review:** use [architecture/status](architecture-status.md) and the dated reports below. Configuration, job completion, optimizer updates, verifier success, and held-out improvement are different claims.

The current hosted workflow uses the Agent Collabs frontend and PostTrain's headless CLI/API. Start with the authorized Space's `/AGENTS.md`, linked from the [HF cookbook](https://posttrain.com/docs/cookbook), and read [hosted collaboration and training status](hf-submission-lab.md). Environment registration and experiment metadata do not launch compute. The checked-in OpenCode/TRL pipeline is a separate implementation.

The [submitted shift-schedule profile](hf-submission-lab.md#recorded-evidence) now has a completed 50-step HF run, saved-adapter reload, independent verifier replay, and reviewed public result: 8/9 → 9/9 checks on one seen task. This does not establish held-out performance or validate the separate full OpenCode/TRL recipe.

## Current guides

| Document | Audience | Purpose |
|---|---|---|
| [`../README.md`](../README.md) | Everyone | Competition overview, repository map, and implementation status |
| [`architecture-status.md`](architecture-status.md) | Everyone | Canonical architecture, ownership boundaries, compatibility matrix, and roadmap |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Contributors | Submission rules, environment authoring, reviews, and pipeline contributions |
| [`training-pipeline.md`](training-pipeline.md) | Organizers and researchers | Canonical BenchFlow + TRL operator guide, configuration, execution, artifacts, and evidence limits |
| [`hf-jobs.md`](hf-jobs.md) | Organizers and Hugging Face collaborators | Submission-to-recipe bridge, HF UV Jobs, artifact publication, multi-benchmark evaluation, and leaderboard hosting |
| [`hf-submission-lab.md`](hf-submission-lab.md) | Everyone | Agent Collabs onboarding, headless submissions, reviewed groups, reservations, and execution evidence |
| [`opencode-grpo.md`](opencode-grpo.md) | Operators | Checked-in OpenCode/TRL rollout contract |
| [`../starting-kit/README.md`](../starting-kit/README.md) | Environment authors | Task package template and worked examples |
| [`../submissions/README.md`](../submissions/README.md) | Teams | Team-entry layout and submission manifest |
| [`../SECURITY.md`](../SECURITY.md) | Security reporters | Private vulnerability reporting and secret-handling expectations |
| [`../SUPPORT.md`](../SUPPORT.md) | Users | Where to ask usage, competition, and incident questions |
| [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Contributors | Community participation and enforcement expectations |

The final competition budget and sealed evaluation remain draft. The public
reference recipe uses the pinned 2,238-task train and 366-task eval datasets;
competition recipes replace those with a participant training corpus and the
organizer's internal evaluation tasks. The Qwen3.5-9B implementation under
`pipelines/benchflow-task-posttrain/` defines executable behavior, while
[`architecture-status.md`](architecture-status.md) defines compatibility and
roadmap status. OpenEnv and HF Jobs are implemented by the public pipeline. An
exploratory 16-train/14-eval same-domain run observed `8/14 → 11/14`; full
public-reference execution, competition generalization, and private-final
evidence remain pending.

## Historical experiments and provider-specific notes

These reports preserve the setup and results of individual experiments. They are not the current submission workflow or an account-capacity check. The current lab trains on Hugging Face; Fireworks notes describe a separate research path.

| Document | Audience | Scope |
| --- | --- | --- |
| [`hf-jobs-validation.md`](hf-jobs-validation.md) | Reviewers and operators | Historical H100 wrapper evidence, Hub outputs, live Space, and the July 11 HF Jobs credit blocker |
| [`fireworks-deployment-safety.md`](fireworks-deployment-safety.md) | Fireworks evaluation operators | Billing-aware deployment setup, scale-to-zero verification, emergency shutdown, and preflight/post-run checks |
| [`from-benchmark-data-to-fireworks-training.md`](from-benchmark-data-to-fireworks-training.md) | Fireworks training researchers | DeepSeek V4 Pro no-skills corpus selection, Qwen3.6-27B overfitting experiment, and pass@10 evaluation |
| [`opencode-evaluation-canary.md`](opencode-evaluation-canary.md) | Reviewers and operators | Historical single-task OpenCode evaluator evidence |
| [`opencode-grpo-smoke.md`](opencode-grpo-smoke.md) | Reviewers and operators | Historical Qwen3-4B OpenCode SFT-to-GRPO plumbing smoke |
| [`native-dataset-openenv-smoke.md`](native-dataset-openenv-smoke.md) | Reviewers and operators | Historical native-dataset OpenEnv execution evidence |
| [`qwen35-opencode-teacher-canary.md`](qwen35-opencode-teacher-canary.md) | Reviewers and operators | Historical single-task Qwen3.5-397B-A17B OpenCode rollout, trajectory, and TRL conversion evidence |
| [`qwen35-data-agent-e2e-canary.md`](qwen35-data-agent-e2e-canary.md) | Reviewers and operators | Qwen3.5-9B LoRA SFT, OpenCode GRPO, synchronization, and exploratory same-domain score evidence |
