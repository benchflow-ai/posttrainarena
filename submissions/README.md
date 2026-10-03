# Team submissions

<!-- markdownlint-disable MD013 MD060 -->

The hosted Arena accepts **1–200 environment packages** from a public GitHub repository or public, ungated HF dataset. Each collection contains a flat `submission.yaml` and `envs/`. The Arena validates its structure and pins its source revision without executing repository code, through its CLI or its Submit a collection form. Start with [submitting a collection and running it](../docs/hf-submission-lab.md).

Submitting registers the collection; it does not launch training. A collection trains on a challenge, which fixes the base model, the post-training recipe and a sealed held-out suite: preflight a run with `arena_cli.py run --challenge`, launch it only with explicit authorization to spend compute, follow it with `runs --run-id`, and collect the scored result with `result collect`. The score is the held-out change (pass rate after training minus before, measured in the same run), and a result ranks after a BenchFlow editor accepts it. There is no 50-package minimum or phase freeze.

Results recorded before challenges, such as the [shift-schedule run](../docs/hf-submission-lab.md#recorded-evidence) (8/9 to 9/9 checks on one seen task), came from the retired experiment path and are seen-task practice, not held-out scores.

## Existing local checker behavior

`check_submission.py` checks the same 1–200 range as the Arena: it rejects an environments entry with no package or more than 200. It also recognizes legacy `track: skills` packages, warning below 20 and rejecting more than 100; that format does not establish a current hosted skill track or evaluator.

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
track: environments
```

The Hugging Face account that submits the collection is its contact. An older manifest's `contact_email` line is accepted but not required.

The local submission checker ignores entry directory names starting with `_`. Use an ordinary entry directory for a contribution; this local convention is not a hosted validation bypass.

## Validating locally

In order: the manifest, bounds and structure of every entry; the same for one collection (any folder with `submission.yaml` and `envs/`); its packages; the oracle replay of one package; and the same package with the oracle skipped:

```bash
python3 scripts/check_submission.py
python3 scripts/check_submission.py submissions/<team-entry>
python3 scripts/check_task.py submissions/<team-entry>/envs
scripts/run_local.sh submissions/<team-entry>/envs/<env>
scripts/run_local.sh submissions/<team-entry>/envs/<env> --skip-oracle
```

Each environment package follows the same contract as the
[starting-kit examples](../starting-kit/examples) — start from
[`starting-kit/template/`](../starting-kit/template) and see
[CONTRIBUTING.md](../CONTRIBUTING.md) for the full walkthrough,
validation ladder, and reviewer checklist.

Use the Arena's CLI to validate and submit your pinned collection, then preflight a run on a challenge; see [submitting a collection and running it](../docs/hf-submission-lab.md). The manifest and each task's credit metadata (author, license, category and origin) provide attribution. Structural validation and local replay are different evidence, and neither queues a hosted job.

## Separate checked-in pipeline preparation

The public Qwen3.5/OpenCode pipeline has a `prepare-submission` utility for operators who run the pipeline themselves. The hosted Arena does not need it: a challenge run snapshots the submitted collection itself. After installing that pipeline, an operator can upload an environment entry and emit a pinned recipe:

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
