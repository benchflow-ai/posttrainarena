# PostTrain Arena — starting kit

<!-- markdownlint-disable MD060 -->

This directory defines the participant-facing **PostTrain task package** format.
It is intentionally runnable with Python and Docker and does not require
BenchFlow, TRL, or OpenEnv for authoring checks.

The public BenchFlow + TRL pipeline in this repository trains on packages in this format: in the hosted Arena, a run on a challenge trains on a submitted collection's eligible tasks. Submitting a collection does not start training; a run is preflighted and launched separately (see [submitting a collection and running it](../docs/hf-submission-lab.md)). Authors do not implement OpenEnv in their packages; the organizer can expose snapshotted BenchFlow tasks through the separate `openenv-serve` compatibility service. See [`docs/architecture-status.md`](../docs/architecture-status.md) for the exact boundary.

Every directory under [`examples/`](./examples) is one **example task
package**, authored
by the organizing team to exercise the `task.md` contract (frontmatter
limits + prompt, an `environment/` Dockerfile with seed data, a
pytest-based `verifier/`, and an `oracle/` that produces a passing
trial). These are reference material for the starting kit — they are
**not automatically registered submissions**. Hosted collections contain 1–200
environments. The [`submissions/`](../submissions) guide explains the manifest,
local checker warnings, and CLI registration.

On the retired experiment path, a pinned public dataset copy of the shift-schedule example completed a hosted seen-task run with independent verifier replay ([recorded evidence](../docs/hf-submission-lab.md#recorded-evidence)). That evidence applies to the linked source commit, not every example or later local edits.

The full authoring reference lives at <https://posttrain.com/docs/spec>;
this README is a short index to what is here.

## Examples

| Task | Author | Category | Origin | License | Difficulty |
|---|---|---|---|---|---|
| [`dogfood-hello-text`](./examples/dogfood-hello-text) | Xiangyi Li | software-engineering | original | AGPL-3.0-only | easy |
| [`skillsbench-3d-scan-calc`](./examples/skillsbench-3d-scan-calc) | Wengao Ye | scientific-computing | adapted | Apache-2.0 | hard |
| [`skillsbench-citation-check`](./examples/skillsbench-citation-check) | Xuandong Zhao | personal-assistant | adapted | Apache-2.0 | medium |
| [`skillsbench-weighted-gdp-calc`](./examples/skillsbench-weighted-gdp-calc) | Xiangyi Li | data-processing | adapted | Apache-2.0 | medium |
| [`seclog-bruteforce-triage`](./examples/seclog-bruteforce-triage) | Xiangyi Li | security | original | AGPL-3.0-only | medium |
| [`subtitle-overlap-qc`](./examples/subtitle-overlap-qc) | Xiangyi Li | video-processing | original | AGPL-3.0-only | medium |
| [`sensor-calibration-fit`](./examples/sensor-calibration-fit) | Xiangyi Li | scientific-computing | original | AGPL-3.0-only | medium |
| [`shift-schedule-verify`](./examples/shift-schedule-verify) | Xiangyi Li | optimization | original | AGPL-3.0-only | medium |

The three `skillsbench-*` tasks were ported from
[SkillsBench](https://skillsbench.ai) as the first reference set; the
rest were authored while dogfooding the submission flow.

Each task declares who wrote it, under what license, what kind of task it is and where it came from, in the `metadata:` block of `task.md`: `author_name`, `author_email`, `license` (an SPDX identifier), `category` and `origin` (`original`, `adapted` with an `origin_url`, or `generated`). `category` is one of the 18 values the Arena's validator accepts, which follow Terminal-Bench 2's categories: `software-engineering`, `system-administration`, `security`, `scientific-computing`, `data-science`, `data-processing`, `data-querying`, `file-operations`, `debugging`, `machine-learning`, `model-training`, `mathematics`, `optimization`, `games`, `personal-assistant`, `video-processing`, `tool-use`, `other`. The Arena's validation warns, without blocking, about a missing or invalid field; the fields are how contributors are credited and how results are analysed by category. The three `skillsbench-*` ports are adapted from SkillsBench and keep its Apache-2.0 license; the other examples are original and use this repository's AGPL-3.0-only. `scripts/check_task.py` checks structure only, not these values.

## Authoring your own

1. Copy [`template/`](./template) into your team entry under
   `submissions/<your-team>/envs/<your-env-name>/`.
2. Fill in `task.md` (including `license`, `category` and `origin`), `environment/Dockerfile` and any seed data, `verifier/test_outputs.py`, and `oracle/solve.sh`.
3. Validate: `python3 scripts/check_task.py <your envs dir>`, then
   `scripts/run_local.sh <your env>` (oracle replay must score 1.0)
   and `scripts/run_local.sh <your env> --skip-oracle` (empty trial
   must not).
4. Validate and submit the pinned public collection with the Arena's CLI, then preflight a run on a challenge ([how](../docs/hf-submission-lab.md)); or open a pull request to contribute examples or tooling to this repository.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the submission model
(current collection bounds and hosted/local boundaries) and the review checklist.

## Naming

`<env-or-domain>-<short-description>` — for example
`skillsbench-weighted-gdp-calc`, `finance-fed-minutes-classify`,
`gmail-workflow-delegation`. Category, modality, and any safety
qualifier live in the frontmatter, not the directory name.
