# PostTrain Arena

The open arena for post-training. Contribute an evaluation task as an
RL-ready environment; we post-train a model across the accepted pool
and score what generalizes.

Site and docs: <https://posttrain.com> · Discord:
<https://discord.gg/mZ9Rc8q8W3>

## What is in this repo

- [`tasks/`](./tasks) — the task pool. Every directory is one
  submission: a `task.md` (frontmatter + prompt), an `environment/`
  Dockerfile with seed data, a mechanical `verifier/`, and an
  `oracle/` that produces a passing trial. See
  [`tasks/README.md`](./tasks/README.md) for the current pool.
- [`tasks/template/`](./tasks/template) — the starting point for a new
  submission.
- [`scripts/check_task.py`](./scripts/check_task.py) — fast
  self-contained structural check (the same one CI runs first).
- [`.github/workflows/tasks-check.yml`](./.github/workflows/tasks-check.yml)
  — CI: structural → schema → publication-grade on changed tasks.

## Contributing a task

The short version:

```bash
cp -R tasks/template tasks/your-task-name
# fill in task.md, environment/, verifier/, oracle/
python3 scripts/check_task.py
bench tasks check ./tasks/your-task-name --level publication-grade
```

Then open a pull request with the tail of the publication-grade output
in the description. The full walkthrough, reviewer checklist, and CI
notes live in [CONTRIBUTING.md](./CONTRIBUTING.md); the authoring
reference is at <https://posttrain.com/docs/spec>.

## License

[AGPL-3.0](./LICENSE). Accepted tasks are released openly as part of
the pool.
