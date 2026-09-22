# PostTrain Arena — starting kit

<!-- markdownlint-disable MD060 -->

This directory defines the participant-facing **PostTrain task package** format.
It is intentionally runnable with Python and Docker and does not require
BenchFlow, TRL, or OpenEnv for authoring checks.

The separate public BenchFlow + TRL pipeline can consume selected task packages.
Hosted training requires a supported execution profile; package authoring or registration does not automatically start training. Authors do not implement OpenEnv in their packages;
the organizer can expose snapshotted BenchFlow tasks through the separate
`openenv-serve` compatibility service. See
[`docs/architecture-status.md`](../docs/architecture-status.md) for the exact
boundary.

Every directory under [`examples/`](./examples) is one **example task
package**, authored
by the organizing team to exercise the `task.md` contract (frontmatter
limits + prompt, an `environment/` Dockerfile with seed data, a
pytest-based `verifier/`, and an `oracle/` that produces a passing
trial). These are reference material for the starting kit — they are
**not automatically registered submissions**. Hosted collections contain 1–200
environments. The [`submissions/`](../submissions) guide explains the manifest,
local checker warnings, and CLI registration.

The full authoring reference lives at <https://posttrain.com/docs/spec>;
this README is a short index to what is here.

## Examples

| Task | Author | Category | Difficulty |
|---|---|---|---|
| [`dogfood-hello-text`](./examples/dogfood-hello-text) | Xiangyi Li | software-engineering | easy |
| [`skillsbench-3d-scan-calc`](./examples/skillsbench-3d-scan-calc) | Wengao Ye | industrial-physical-systems | hard |
| [`skillsbench-citation-check`](./examples/skillsbench-citation-check) | Xuandong Zhao | office-white-collar | medium |
| [`skillsbench-weighted-gdp-calc`](./examples/skillsbench-weighted-gdp-calc) | Xiangyi Li | finance-economics | medium |
| [`seclog-bruteforce-triage`](./examples/seclog-bruteforce-triage) | Xiangyi Li | cybersecurity | medium |
| [`subtitle-overlap-qc`](./examples/subtitle-overlap-qc) | Xiangyi Li | media-content-production | medium |
| [`sensor-calibration-fit`](./examples/sensor-calibration-fit) | Xiangyi Li | industrial-physical-systems | medium |
| [`shift-schedule-verify`](./examples/shift-schedule-verify) | Xiangyi Li | mathematics-or-formal-reasoning | medium |

The three `skillsbench-*` tasks were ported from
[SkillsBench](https://skillsbench.ai) as the first reference set; the
rest were authored while dogfooding the submission flow.

The examples use SkillsBench category slugs in task frontmatter. Follow the current authoring specification; structural checks alone do not validate every vocabulary value or establish compatibility with an executor.

## Authoring your own

1. Copy [`template/`](./template) into your team entry under
   `submissions/<your-team>/envs/<your-env-name>/`.
2. Fill in `task.md`, `environment/Dockerfile` and any seed data,
   `verifier/test_outputs.py`, and `oracle/solve.sh`.
3. Validate: `python3 scripts/check_task.py <your envs dir>`, then
   `scripts/run_local.sh <your env>` (oracle replay must score 1.0)
   and `scripts/run_local.sh <your env> --skip-oracle` (empty trial
   must not).
4. Register the pinned public collection through the [hosted CLI/API](../docs/hf-submission-lab.md), or open a pull request to contribute examples or tooling to this repository.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the submission model
(current collection bounds and hosted/local boundaries) and the review checklist.

## Naming

`<env-or-domain>-<short-description>` — for example
`skillsbench-weighted-gdp-calc`, `finance-fed-minutes-classify`,
`gmail-workflow-delegation`. Category, modality, and any safety
qualifier live in the frontmatter, not the directory name.
