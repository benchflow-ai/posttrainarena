# PostTrain Arena — tasks

Every directory at this level is one submission. The full authoring
reference lives at <https://posttrain.com/docs/spec>; this README is a
short index to what is here.

## Current pool

| Task | Author | Category | Difficulty |
|---|---|---|---|
| [`dogfood-hello-text`](./dogfood-hello-text) | Xiangyi Li | software-engineering | easy |
| [`skillsbench-3d-scan-calc`](./skillsbench-3d-scan-calc) | Wengao Ye | industrial-physical-systems | hard |
| [`skillsbench-citation-check`](./skillsbench-citation-check) | Xuandong Zhao | office-white-collar | medium |
| [`skillsbench-weighted-gdp-calc`](./skillsbench-weighted-gdp-calc) | Xiangyi Li | finance-economics | medium |
| [`seclog-bruteforce-triage`](./seclog-bruteforce-triage) | Xiangyi Li | cybersecurity | medium |
| [`subtitle-overlap-qc`](./subtitle-overlap-qc) | Xiangyi Li | media-content-production | medium |
| [`sensor-calibration-fit`](./sensor-calibration-fit) | Xiangyi Li | industrial-physical-systems | medium |
| [`shift-schedule-verify`](./shift-schedule-verify) | Xiangyi Li | mathematics-or-formal-reasoning | medium |

All three were ported from [SkillsBench](https://skillsbench.ai) as the
first reference set — they exercise every part of the `task.md` contract
(frontmatter limits + prompt, an `environment/` Dockerfile with seed
data and bundled skills, a pytest-based `verifier/`, and an `oracle/`
that produces a passing trial).

## Adding a task

1. Copy [`template/`](./template) to `tasks/your-task-name/`.
2. Fill in `task.md`, `environment/Dockerfile` and any seed data,
   `verifier/test_outputs.py`, and `oracle/solve.sh`.
3. Run `bench tasks check ./tasks/your-task-name --level publication-grade`
   until everything passes.
4. Open a pull request.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full review checklist.

## Naming

`<env-or-domain>-<short-description>` — for example
`skillsbench-weighted-gdp-calc`, `finance-fed-minutes-classify`,
`gmail-workflow-delegation`. Category, modality, and any safety
qualifier live in the frontmatter, not the directory name.
