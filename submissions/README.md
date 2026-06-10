# Team submissions

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
python3 scripts/check_submission.py            # checks submissions/
python3 scripts/check_task.py submissions/<team-entry>/envs
```

Each environment package follows the same contract as the
[starting-kit examples](../tasks/) — start from
[`tasks/template/`](../tasks/template) and see
[CONTRIBUTING.md](../CONTRIBUTING.md) for the full walkthrough,
validation ladder, and reviewer checklist.

Entries are graded blind to author identity; the `author_*` frontmatter
fields are used for credit after scoring, not during it. Teams may
withdraw an entry until the Phase 2 freeze.
