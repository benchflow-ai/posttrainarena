# PostTrain Arena

A proposed NeurIPS 2026 competition on agentic RL environments across
diverse domains beyond coding. PostTrain Arena inverts the usual
contract: **teams contribute containerized RL environments**, and the
organizers run a managed SFT→GRPO post-training pipeline on each
team's corpus. The resulting Qwen3-8B checkpoint is scored on a
private held-out suite (BenchFlow Signals), and entries are ranked by
the held-out generalization delta over a fixed reference baseline.

Site and docs: <https://posttrain.com> · Discord:
<https://discord.gg/mZ9Rc8q8W3> · Contact: labs@benchflow.ai

## The competition at a glance

- **Submissions are bounded by teams.** Track 2 — Environment
  Submission (headline): 50 min / 100 recommended / 200 max
  environments per entry, full managed pipeline, compute sponsored.
  Track 1 — Skill Learning (low barrier): 20/50/100 `SKILL.md`
  packages per entry, evaluated by pass@1 of a frozen reference agent.
  Teams may enter both tracks as separate entries.
- **Scoring.** Δ = pass-rate of the checkpoint trained on your corpus
  minus the fixed reference baseline, on a sealed 100-task suite with
  paired bootstrap confidence intervals. A 20-task public sample is
  released for sanity checks.
- **Phases.** Phase 0 warm-up (public sample only) → Phase 1
  development (live public-sample feedback) → Phase 2 final
  (submissions frozen, private-suite evaluation). Withdrawal allowed
  until the Phase 2 freeze; grading is blind to author identity.
- **Licensing (draft rules).** Submissions: CC-BY-4.0 (text/data) +
  Apache-2.0 (code); authorship retained. Accepted environments,
  teacher data, and trained checkpoints are released openly after the
  competition.

Status: proposal under review; rules are draft until the starting kit
ships. This repo is the starting-kit preview and the future home of
team submissions.

## What is in this repo

- [`tasks/`](./tasks) — organizer-authored **starting-kit examples**
  exercising the full task contract (`task.md` + `environment/` +
  `verifier/` + `oracle/`). Not competition entries.
- [`submissions/`](./submissions) — the team submission tree: one
  directory per team entry with a `submission.yaml` manifest. See its
  README for layout and bounds.
- [`scripts/check_task.py`](./scripts/check_task.py) and
  [`scripts/check_submission.py`](./scripts/check_submission.py) —
  fast self-contained local checks (the same ones CI runs first).
- [`.github/workflows/tasks-check.yml`](./.github/workflows/tasks-check.yml)
  — CI: structural + submission bounds → schema → publication-grade.
  The gauntlet grows to oracle execution and leakage audits before
  Phase 0.

## Quick start

```bash
mkdir -p submissions/your-team/envs
cp -R tasks/template submissions/your-team/envs/your-env-name
# fill in task.md, environment/, verifier/, oracle/
# add submissions/your-team/submission.yaml (see submissions/README.md)
python3 scripts/check_task.py submissions/your-team/envs
python3 scripts/check_submission.py
```

Then see [CONTRIBUTING.md](./CONTRIBUTING.md) for the submission
model, validation ladder, and reviewer checklist. The authoring
reference is at <https://posttrain.com/docs/spec>.

## License

Repository contents: [AGPL-3.0](./LICENSE) unless noted otherwise.
Competition submissions are licensed by their authors under CC-BY-4.0
(text/data) + Apache-2.0 (code) at submission time, per the draft
rules.
