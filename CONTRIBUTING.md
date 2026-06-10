# Contributing to PostTrain Arena

PostTrain Arena is a proposed NeurIPS 2026 competition: teams
contribute containerized RL environments, the organizers run a managed
SFT→GRPO post-training pipeline on **each team's corpus**, and entries
are ranked by the held-out generalization delta of the resulting
checkpoint. This repo hosts the starting kit (the task template and
worked examples under [`starting-kit/`](./starting-kit)) and the team
submission tree ([`submissions/`](./submissions)).

The full authoring reference lives at
<https://posttrain.com/docs/spec>; this file is the short version with
links to the right places.

## How submission works

**Submissions are bounded by teams.** The unit of entry is a team's
corpus on one track, not an individual task:

| Track | Package | Min | Recommended | Max per entry |
|---|---|---|---|---|
| Track 2 — Environment Submission | task package (Docker + verifier + oracle + instruction) | 50 | 100 | 200 |
| Track 1 — Skill Learning | `SKILL.md` package | 20 | 50 | 100 |

Teams may enter both tracks as separate entries. One entry is one
directory under [`submissions/`](./submissions) with a
`submission.yaml` manifest — see that README for the layout.

**Scoring (Track 2, headline).** The managed pipeline regenerates
teacher trajectories, runs SFT and GRPO on Qwen3-8B over your
environments, and evaluates the checkpoint on BenchFlow Signals — a
private 100-task held-out suite (a 20-task public sample is released
for sanity checks). Your score is the delta over a fixed reference
baseline trained with the identical recipe, with paired bootstrap
confidence intervals. Track 1 packages are evaluated by pass@1 of a
frozen reference agent — no training, no internet.

**Phases.** Phase 0 (warm-up): public sample only, leaderboard hidden.
Phase 1 (development): full entries accepted, public-sample scoring
shown live. Phase 2 (final): submissions frozen, private-suite
evaluation. Entries may be withdrawn until the Phase 2 freeze, and
grading is blind to author identity.

**Licensing (draft rules, finalized in the starting kit).** Submissions
are licensed CC-BY-4.0 (text/data) + Apache-2.0 (code) at submission
time; participants retain authorship. Accepted environments, teacher
data, and trained checkpoints are released openly after the
competition; teams may flag individual environments as
"release-only, training-excluded".

## Authoring an environment

Every environment package is one directory with four parts: `task.md`,
`environment/`, `verifier/`, `oracle/`. The
[task template](./starting-kit/template) is the fastest way to start;
the worked examples under
[`starting-kit/examples/`](./starting-kit/examples) exercise every
part of the contract.

### Step-by-step

1. **Copy the template** into your team entry:
   ```bash
   cp -R starting-kit/template submissions/your-team/envs/your-env-name
   ```
   Pick a name following `<env-or-domain>-<short-description>` — for
   example `gmail-workflow-delegation`. Category, modality, and any
   safety qualifier belong in the frontmatter, not the directory name.
2. **Fill in the four parts.** Read the
   [spec](https://posttrain.com/docs/spec) for the full reference; the
   [`starting-kit/examples/`](./starting-kit/examples) show real
   layouts.
3. **Validate locally** — everything runs with just python3 and
   docker, no benchflow install:
   ```bash
   # Structural — fast, no Docker required
   python3 scripts/check_task.py submissions/your-team/envs
   python3 scripts/check_submission.py

   # Oracle replay — build the image, run your oracle, score it
   scripts/run_local.sh submissions/your-team/envs/your-env-name

   # Empty trial — prove the verifier rejects a do-nothing run
   scripts/run_local.sh submissions/your-team/envs/your-env-name --skip-oracle
   ```
   Get all four green before opening a PR: the oracle replay must
   score 1.0 and the empty trial must not.
4. **Open a pull request** adding or updating your team entry. In the
   description, paste the tail of both `run_local.sh` runs.

### What reviewers check

- **Schema.** Frontmatter validates; required fields present; tags and
  category are from the published vocabulary (see the spec).
- **Build.** `environment/` builds inside the budget you declared.
- **Solvability.** `oracle/solve.sh` produces a passing trial under the
  same image. The oracle is documentation as much as a regression
  check — keep it as simple as the task allows.
- **Verifier sanity.** `verifier/test_outputs.py` distinguishes real
  trial output from trivially-empty output and from the oracle's exact
  bytes. A verifier that only checks for a fixed file is too weak.
- **Network.** The environment's network policy matches what the task
  actually needs; opt in only when the task requires the public web.
- **Robustness.** Resistance to reward hacking, prompt injection, and
  verifier shortcuts is a first-class review criterion — expect
  adversarial probing of your verifier.

Before Phase 0 the CI gauntlet grows to match the competition
protocol: structural validation, oracle execution, instruction-quality
screening, a leakage audit against the public sample, and a 3-stage
Docker/verifier/difficulty filter. Accepted entries join the training
queue.

## Contributing to the landing page

The site at <https://posttrain.com> is developed in a separate
repository. For site bugs or copy fixes, open an issue here or ping us
on Discord and we will route it.

## Maintainer notes

### CI: `tasks-check` workflow

`.github/workflows/tasks-check.yml` runs two fully self-contained
jobs — no secrets, no private dependencies, so fork PRs get exactly
the same checks as everyone else:

1. **structural** — `scripts/check_task.py`, ~1s, no external deps.
2. **submissions** — `scripts/check_submission.py`: team manifest,
   track bounds (warn below min until the Phase 2 freeze, fail above
   max), per-package structure.

Everything deeper — schema validation, oracle execution,
instruction-quality screening, the leakage audit — runs in the managed
pipeline after a PR is opened, and locally via
`scripts/run_local.sh` (docker only, no benchflow install).

## Getting help

- Discord: <https://discord.gg/mZ9Rc8q8W3>
- Spec questions: open a discussion thread on this repo.
