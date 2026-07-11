# Team submissions

Submissions use the PostTrain Arena task-package and team-manifest contract.
They are validated locally with the self-contained scripts in this repository;
organizer training uses BenchFlow + TRL after intake. OpenEnv compatibility is
not required in a team package; the organizer pipeline supplies the implemented
OpenEnv protocol integration.
See [`docs/architecture-status.md`](../docs/architecture-status.md).

A competition entry is **one directory here, owned by one team, on one
track**. Submissions are bounded per team — you submit a corpus, not
individual tasks, and the managed pipeline trains on your corpus as a
unit.

| Track | Package | Min | Recommended | Max |
|---|---|---|---|---|
| `environments` | task package (`task.md` + `environment/` + `verifier/` + `oracle/`) | 50 | 100 | 200 |
| `skills` | `SKILL.md` package | 20 | 50 | 100 |

Teams may enter both tracks as **separate entries** (two directories).
Minimums are enforced at the Phase 2 freeze, not before — CI warns
below the minimum and fails above the maximum. The published rules
allow lowering the environments minimum if participation is low.

## Layout

```
submissions/<team-entry>/
  submission.yaml          # flat key: value
  envs/<env-name>/...      # track: environments
  skills/<skill-name>/...  # track: skills
```

`submission.yaml` (flat keys only, no nesting):

```yaml
team_name: Your Team
contact_email: you@example.com
track: environments        # or: skills
```

Directory names starting with `_` are ignored by CI (scratch space).

## Validating locally

```bash
python3 scripts/check_submission.py            # manifest + bounds + structure
python3 scripts/check_task.py submissions/<team-entry>/envs
scripts/run_local.sh submissions/<team-entry>/envs/<env>   # oracle replay
```

Each environment package follows the same contract as the
[starting-kit examples](../starting-kit/examples) — start from
[`starting-kit/template/`](../starting-kit/template) and see
[CONTRIBUTING.md](../CONTRIBUTING.md) for the full walkthrough,
validation ladder, and reviewer checklist.

Entries are graded blind to author identity; the `author_*` frontmatter
fields are used for credit after scoring, not during it. Teams may
withdraw an entry until the Phase 2 freeze.

## Organizer handoff

After intake, an organizer can upload an environment entry and emit a pinned
training recipe:

```bash
posttrainarena-train prepare-submission \
  --entry submissions/<team-entry> \
  --base-config pipelines/benchflow-task-posttrain/configs/qwen3-4b-data-agent-openenv-smoke.toml \
  --out .local/prepared/<team-entry> \
  --dataset-repo <namespace>/posttrainarena-<team-entry> \
  --upload
```

This managed SFT/GRPO bridge supports `track: environments`. Skill-track
evaluation remains a separate frozen-agent workflow.
