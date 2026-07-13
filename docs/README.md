# Documentation map

| Document | Audience | Purpose |
|---|---|---|
| [`../README.md`](../README.md) | Everyone | Competition overview, repository map, and implementation status |
| [`architecture-status.md`](architecture-status.md) | Everyone | Canonical architecture, ownership boundaries, compatibility matrix, and roadmap |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Contributors | Submission rules, environment authoring, reviews, and pipeline contributions |
| [`training-pipeline.md`](training-pipeline.md) | Organizers and researchers | Canonical BenchFlow + TRL operator guide, configuration, execution, artifacts, and evidence limits |
| [`hf-jobs.md`](hf-jobs.md) | Organizers and Hugging Face collaborators | Submission-to-recipe bridge, HF UV Jobs, artifact publication, multi-benchmark evaluation, and leaderboard hosting |
| [`hf-jobs-validation.md`](hf-jobs-validation.md) | Reviewers and operators | H100 wrapper evidence, Hub outputs, live Space, and the current HF Jobs credit blocker |
| [`qwen35-opencode-teacher-canary.md`](qwen35-opencode-teacher-canary.md) | Reviewers and operators | Real Qwen3.5-397B OpenCode rollout, trajectory, and TRL conversion evidence |
| [`../starting-kit/README.md`](../starting-kit/README.md) | Environment authors | Task package template and worked examples |
| [`../submissions/README.md`](../submissions/README.md) | Teams | Team-entry layout and submission manifest |
| [`../SECURITY.md`](../SECURITY.md) | Security reporters | Private vulnerability reporting and secret-handling expectations |
| [`../SUPPORT.md`](../SUPPORT.md) | Users | Where to ask usage, competition, and incident questions |

The final competition budget and sealed evaluation remain draft. The
Qwen3.5-9B implementation under `pipelines/benchflow-task-posttrain/` defines
executable behavior, while
[`architecture-status.md`](architecture-status.md) defines compatibility and
roadmap status. OpenEnv and HF Jobs are implemented by the public pipeline;
competition-scale recipes and private final evaluation remain draft.
