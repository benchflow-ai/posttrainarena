# PostTrain Arena — starting kit

Every directory under [`examples/`](./examples) is one **example task
package**, authored
by the organizing team to exercise the `task.md` contract (frontmatter
limits + prompt, an `environment/` Dockerfile with seed data, a
pytest-based `verifier/`, and an `oracle/` that produces a passing
trial). These are reference material for the starting kit — they are
**not competition entries**. Team entries are corpora of 50–200
environments and live under [`submissions/`](../submissions).

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

A note on vocabulary: the `category` slugs in task frontmatter follow
the SkillsBench taxonomy. The competition's public domain list uses
display names (Sciences, Industrial & Energy Operations, …); the
authoritative slug↔domain mapping will be finalized in the competition
white-paper and starting kit.

## Authoring your own

1. Copy [`template/`](./template) into your team entry under
   `submissions/<your-team>/envs/<your-env-name>/`.
2. Fill in `task.md`, `environment/Dockerfile` and any seed data,
   `verifier/test_outputs.py`, and `oracle/solve.sh`.
3. Validate: `python3 scripts/check_task.py <your envs dir>`, then
   `scripts/run_local.sh <your env>` (oracle replay must score 1.0)
   and `scripts/run_local.sh <your env> --skip-oracle` (empty trial
   must not).
4. Open a pull request.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the submission model
(tracks, per-team bounds, phases) and the full review checklist.

## Naming

`<env-or-domain>-<short-description>` — for example
`skillsbench-weighted-gdp-calc`, `finance-fed-minutes-classify`,
`gmail-workflow-delegation`. Category, modality, and any safety
qualifier live in the frontmatter, not the directory name.
