# Documentation map

| Document | Audience | Purpose |
|---|---|---|
| [`../README.md`](../README.md) | Everyone | Competition overview, repository map, and implementation status |
| [`architecture-status.md`](architecture-status.md) | Everyone | Canonical architecture, ownership boundaries, compatibility matrix, and roadmap |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Contributors | Submission rules, environment authoring, reviews, and pipeline contributions |
| [`training-pipeline.md`](training-pipeline.md) | Organizers and researchers | Canonical BenchFlow + TRL operator guide, configuration, execution, artifacts, and evidence limits |
| [`../starting-kit/README.md`](../starting-kit/README.md) | Environment authors | Task package template and worked examples |
| [`../submissions/README.md`](../submissions/README.md) | Teams | Team-entry layout and submission manifest |
| [`../SECURITY.md`](../SECURITY.md) | Security reporters | Private vulnerability reporting and secret-handling expectations |
| [`../SUPPORT.md`](../SUPPORT.md) | Users | Where to ask usage, competition, and incident questions |

The competition recipe is still draft. The implementation under
`pipelines/benchflow-task-posttrain/` defines executable behavior, while
[`architecture-status.md`](architecture-status.md) defines compatibility and
roadmap status. OpenEnv compatibility and HF Jobs execution are not currently
implemented.
