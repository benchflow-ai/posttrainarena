# Team submissions

<!-- markdownlint-disable MD013 MD060 -->

The current hosted workflow accepts **1–200 environment packages** from a public GitHub repository or public, ungated HF dataset. Each collection contains a flat `submission.yaml` and `envs/`. The headless CLI/API validates its structure and pins its source revision without executing repository code. Start with [hosted collaboration and execution](../docs/hf-submission-lab.md).

Model, training data, evaluation data, method, and parameters are configurable today. Registration records metadata; it does not launch training. Hosted execution requires a supported profile and explicit compute authorization. Participant scores need organizer evidence review, and public-feed publication is separate. Fixed hackathon protocols can be published later; there is no current 50-package minimum, phase-freeze requirement, or universal managed-training guarantee for hosted collections.

## Existing local checker behavior

The self-contained scripts retain some historical competition constraints. `check_submission.py` accepts at least one environment package, warns below 50, and rejects more than 200. Its warning does not override the hosted 1–200 range. It also recognizes legacy `track: skills` packages, warning below 20 and rejecting more than 100; that format does not establish a current hosted skill track or evaluator.

OpenEnv implementation is not required inside a submitted package. The separate checked-in pipeline provides an adapter; see [architecture/status](../docs/architecture-status.md) for that implementation's boundaries.

## Layout

```text
submissions/<team-entry>/
  submission.yaml          # flat key: value
  envs/<env-name>/...      # track: environments
```

`submission.yaml` (flat keys only, no nesting):

```yaml
team_name: Your Team
contact_email: you@example.com
track: environments
```

The local submission checker ignores entry directory names starting with `_`. Use an ordinary entry directory for a contribution; this local convention is not a hosted validation bypass.

## Validating locally

```bash
python3 scripts/check_submission.py            # manifest + bounds + structure
python3 scripts/check_task.py submissions/<team-entry>/envs
scripts/run_local.sh submissions/<team-entry>/envs/<env>   # oracle replay
scripts/run_local.sh submissions/<team-entry>/envs/<env> --skip-oracle
```

Each environment package follows the same contract as the
[starting-kit examples](../starting-kit/examples) — start from
[`starting-kit/template/`](../starting-kit/template) and see
[CONTRIBUTING.md](../CONTRIBUTING.md) for the full walkthrough,
validation ladder, and reviewer checklist.

Use the hosted CLI to validate and submit your pinned collection, then register a matching experiment. The manifest and task author fields provide attribution; current registration does not claim blind competition grading. Structural validation and local replay are different evidence, and neither queues a hosted job.

## Separate checked-in pipeline preparation

The public Qwen3.5/OpenCode pipeline has a `prepare-submission` utility. This is separate from the hosted Agent Collabs registry and its execution profiles. After installing that pipeline, an operator can upload an environment entry and emit a pinned recipe:

```bash
posttrainarena-train prepare-submission \
  --entry submissions/<team-entry> \
  --base-config pipelines/benchflow-task-posttrain/configs/qwen3.5-9b-data-agent-full.toml \
  --out .local/prepared/<team-entry> \
  --dataset-repo <namespace>/posttrainarena-<team-entry> \
  --upload
```

This command uploads data because it includes `--upload`; omit that flag for local preparation. The example inherits the pinned 366-task public evaluation list. Preparation replaces the training dataset/list with the participant corpus and sets strict teacher coverage to that corpus size. It does not execute training or prove an evaluation result.

The utility supports `track: environments`. A different evaluation suite requires a separately reviewed base configuration; no sealed competition protocol or skill evaluator is established by this command.
