# Contributing to PostTrain Arena

Thanks for considering a contribution. Almost everything that lands
here is an evaluation task. The full authoring reference lives at
<https://posttrain.com/docs/spec>; this file is the short version with
links to the right places.

## Contributing a task

A submission is one directory under [`tasks/`](./tasks) with four
parts: `task.md`, `environment/`, `verifier/`, `oracle/`. The
[task template](./tasks/template) is the fastest way to start; the
three [`skillsbench-*`](./tasks) tasks are working references that
exercise every part of the contract.

### Step-by-step

1. **Fork the repo** and `git checkout -b your-task-name`.
2. **Copy the template:**
   ```bash
   cp -R tasks/template tasks/your-task-name
   ```
   Pick a name following `<env-or-domain>-<short-description>` — for
   example `gmail-workflow-delegation`. Category, modality, and any
   safety qualifier belong in the frontmatter, not the directory name.
3. **Fill in the four files.** Read the
   [spec](https://posttrain.com/docs/spec) for the full reference;
   the dogfooded SkillsBench tasks
   ([3d-scan-calc](./tasks/skillsbench-3d-scan-calc),
   [citation-check](./tasks/skillsbench-citation-check),
   [weighted-gdp-calc](./tasks/skillsbench-weighted-gdp-calc))
   show real layouts.
4. **Validate locally:**
   ```bash
   # Schema-only — fast, no Docker required
   bench tasks check ./tasks/your-task-name --level schema

   # Publication-grade — builds the image, runs the oracle, scores it
   bench tasks check ./tasks/your-task-name --level publication-grade
   ```
   Get to a clean publication-grade run before opening a PR.
5. **Open a pull request.** In the description, paste the tail of the
   `publication-grade` output so reviewers know it built and the oracle
   passed on your machine.

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
- **Network.** `environment.network_mode` matches what the task
  actually needs. Default is `no-network`; opt in only when the task
  requires the public web.
- **Originality.** The task isn't a trivial variant of an existing
  task. We aim for diverse coverage across the eight under-served
  domains, not depth on one.

If acceptance is rejected, the PR comment will say which check failed
and how to fix it.

### What happens after merge

Accepted tasks are released openly. We periodically post-train a
Qwen3-8B model across the accepted pool and score it on the private
IndexBench held-out suite. The standings on the landing page reflect
the lift each task contributed.

## Contributing to the landing page

The site at <https://posttrain.com> is developed in a separate
repository. For site bugs or copy fixes, open an issue here or ping us
on Discord and we will route it.

## Maintainer notes

### CI: `tasks-check` workflow

`.github/workflows/tasks-check.yml` runs three jobs:

1. **structural** — `scripts/check_task.py`, ~1s, no external deps.
2. **schema** — `bench tasks check --level schema` on every task.
3. **publication-grade** — `bench tasks check --level publication-grade --sandbox docker` on PR-changed tasks only.

Stages 2 and 3 install the benchflow CLI from the **private**
`benchflow-ai/benchflow-task-standard-private` repo (branch
`codex/task-md-dogfood-schema-check`), because the `task.md` format and
the multi-level check live on a pre-release dogfood branch.

**Required repo secret:** `BENCHFLOW_TOKEN` — a fine-grained PAT with
read access to `benchflow-ai/benchflow-task-standard-private`
(Contents: read). Configure under *Settings → Secrets and variables →
Actions → New repository secret*. Without it, jobs 2 and 3 fail at
checkout time; job 1 keeps running so PRs still get fast feedback on
the obvious mistakes.

**Known limitation — fork PRs:** GitHub does not expose repository
secrets to workflows triggered by pull requests from forks, so jobs 2
and 3 will fail at the private-repo checkout on every external
contribution. Until the format ships publicly, treat job 1
(structural) as the only required check on fork PRs and rely on the
contributor's pasted local `publication-grade` output plus maintainer
re-runs for the rest.

Once the new format ships on `benchflow-ai/benchflow` main, swap the
two `actions/checkout` blocks for a single
`uv tool install --prerelease=allow 'benchflow @ git+https://github.com/benchflow-ai/benchflow@main'`
step and drop the `BENCHFLOW_TOKEN` secret.

## Getting help

- Discord: <https://discord.gg/mZ9Rc8q8W3>
- Spec questions: open a discussion thread on this repo.
