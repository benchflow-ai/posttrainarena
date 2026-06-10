# Contributing to PostTrain Arena

PostTrain Arena is a proposed NeurIPS 2026 competition: teams
contribute containerized RL environments, the organizers run a managed
SFT→GRPO post-training pipeline on **each team's corpus**, and entries
are ranked by the held-out generalization delta of the resulting
checkpoint. This repo hosts the starting-kit material (the task
template and worked examples under [`tasks/`](./tasks)) and the team
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
[task template](./tasks/template) is the fastest way to start; the
worked examples under [`tasks/`](./tasks) exercise every part of the
contract.

### Step-by-step

1. **Copy the template** into your team entry:
   ```bash
   cp -R tasks/template submissions/your-team/envs/your-env-name
   ```
   Pick a name following `<env-or-domain>-<short-description>` — for
   example `gmail-workflow-delegation`. Category, modality, and any
   safety qualifier belong in the frontmatter, not the directory name.
2. **Fill in the four parts.** Read the
   [spec](https://posttrain.com/docs/spec) for the full reference; the
   [`tasks/`](./tasks) examples show real layouts.
3. **Validate locally:**
   ```bash
   # Structural — fast, no Docker required
   python3 scripts/check_task.py submissions/your-team/envs
   python3 scripts/check_submission.py

   # Schema-only via the benchflow CLI
   bench tasks check ./submissions/your-team/envs/your-env-name --level schema

   # Publication-grade — package-contract check
   bench tasks check ./submissions/your-team/envs/your-env-name --level publication-grade
   ```
   Get to a clean publication-grade run before opening a PR, and prove
   oracle solvability with a live run (e.g.
   `bench eval create --agent oracle --sandbox docker`) — the
   publication-grade gate alone does not execute the oracle.
4. **Open a pull request** adding or updating your team entry. In the
   description, paste the tail of your check output and the oracle-run
   evidence.

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

`.github/workflows/tasks-check.yml` runs four jobs:

1. **structural** — `scripts/check_task.py`, ~1s, no external deps.
2. **submissions** — `scripts/check_submission.py`: team manifest,
   track bounds (warn below min, fail above max), per-package
   structure.
3. **schema** — `bench tasks check --level schema` on every task.
4. **publication-grade** — `bench tasks check --level publication-grade --sandbox docker` on PR-changed tasks only.

Jobs 3 and 4 install the benchflow CLI from the **private**
`benchflow-ai/benchflow-task-standard-private` repo (branch
`codex/task-md-dogfood-schema-check`), because the `task.md` format and
the multi-level check live on a pre-release dogfood branch.

**Required repo secret:** `BENCHFLOW_TOKEN` — a fine-grained PAT with
read access to `benchflow-ai/benchflow-task-standard-private`
(Contents: read). Configure under *Settings → Secrets and variables →
Actions → New repository secret*. Without it, jobs 3 and 4 fail at
checkout time; jobs 1–2 keep running so PRs still get fast feedback on
the obvious mistakes.

**Known limitation — fork PRs:** GitHub does not expose repository
secrets to workflows triggered by pull requests from forks, so jobs 3
and 4 will fail at the private-repo checkout on every external
contribution. Until the format ships publicly, treat jobs 1–2 as the
only required checks on fork PRs and rely on the contributor's pasted
local output plus maintainer re-runs for the rest.

Once the new format ships on `benchflow-ai/benchflow` main, swap the
private clone steps for a single
`uv tool install --prerelease=allow 'benchflow @ git+https://github.com/benchflow-ai/benchflow@main'`
step and drop the `BENCHFLOW_TOKEN` secret.

## Getting help

- Discord: <https://discord.gg/mZ9Rc8q8W3>
- Spec questions: open a discussion thread on this repo.
