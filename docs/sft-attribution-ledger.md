# SFT attribution on agentic evals: experiment ledger

This document records the experiment series behind PostTrain Arena's
attribution pipeline — the instrument that answers, for the competition and for
the OpenEnv integration lane, one question end to end:

> Does *this* model, trained on *this* data mixture, beat its own base on
> *this* eval mixture — by how much, with what confidence interval, under a
> per-domain cap that no single domain can dominate?

Everything below was measured with the same pipeline that scores competition
submissions. Numbers are paired deltas (candidate minus its own base, same
serving stack, same tasks), with bootstrap confidence intervals.

## Method

1. **Corpus build.** Render training rows from environment rollouts or task
   oracle solutions (instruction → solution), with the eval set held out.
2. **Training.** LoRA SFT on the base model (Qwen3.6-27B dense unless noted)
   via a managed fine-tuning API.
3. **Serving.** Both arms are served from one shared deployment: the baseline
   hits the deployment's base model, the candidate hits a LoRA adapter loaded
   onto the same deployment — same hardware, same runtime, so the delta is the
   weights and nothing else.
4. **Paired eval.** Each eval-mixture component runs both arms over identical
   task sets in isolated sandboxes; per-component paired lift with bootstrap
   CIs; components aggregate under a ≤20% per-domain cap.
5. **Two-stage screening.** The eval baseline is noisy (mean reward 0.340,
   sd 0.444 across 12 SkillsBench tasks → a 1-attempt CI is roughly ±0.23
   wide). So every checkpoint is screened at 1 attempt/task, and only the best
   point estimate re-runs at 3 attempts (36 paired observations, half-width
   ≈0.13). Only the confirmation run is reported.

## Result 1 — training works (memorization probe)

Before interpreting any benchmark delta, verify the checkpoint actually
learned its data. Token-F1 on the training rows vs held-out rows, across 8
checkpoints:

| | token-F1 lift |
|---|---|
| training rows | **+0.786** |
| held-out rows | +0.002 |

The pipeline trains and the checkpoints memorize. Benchmark deltas are
therefore about *transfer*, not about broken training.

## Result 2 — five corpora, five negatives, one quantitative law

Six SFT configurations, all paired deltas ≤ 0 on the 12-task SkillsBench
mixture. The mechanism: **inference-time tool use tracks the training
corpus's step count**, against a base model that makes ~30 tool calls per
task:

| corpus (calls/row) | calls at inference | paired delta |
|---|---|---|
| 0 (rendered documents) | 19.0 | −0.400 |
| 4 (terse trajectories) | 4.0 | — |
| 14 (longer trajectories) | 8.6 | −0.358 |
| 39 (agentic trajectories) | 18.7 | −0.165 |

Damage shrinks as corpus step count rises but never reaches zero — even for a
corpus with *more* steps per row than the base model takes. For an agentic
benchmark, **corpus format and step count dominate corpus domain** — the
reverse of the domain-mismatch hypothesis this series started from.

A cautionary replicate: one candidate screened at CI [−0.288, +0.021] and
looked like a statistically reachable hillclimb. Three confirmation replicates
scored 0/3 on all three of the tasks in question: capability loss, not
variance. The confirmation stage exists to catch exactly that, and it did.

## Result 3 — the perturbation law

Damage falls monotonically along two independent axes:

| lever | setting → delta |
|---|---|
| corpus step count (LoRA r32, lr 1e-4) | 0 → −0.400 · 14 → −0.358 · 39 → −0.165 |
| perturbation size (agentic corpus) | r32/1e-4 → −0.165 · **r4/1e-5 → −0.056** |

The minimal perturbation (rank 4, lr 1e-5) on the agentic corpus is the first
candidate whose interval is centred near zero (CI [−0.212, +0.066]) —
statistically indistinguishable from base rather than better than it — and the
first to hold every task the base reliably solves.

**Timeout hypothesis disproven.** The five never-solved tasks time out often
at 900 s, which suggested a budget artifact. Re-run at 2400 s, two of them ran
to completion with no timeout and still scored zero. The benchmark's hard tail
is genuine capability failure; raising budgets does not manufacture headroom.

## Result 4 — the leak control (known-positive instrument check)

Every result above is a negative, which leaves two indistinguishable
explanations: "SFT hurts this benchmark" and "the instrument cannot detect
improvement." The leak control separates them by contamination on purpose:
train on the *evaluated tasks' own* oracle solutions, then evaluate on those
same tasks. If the pipeline can detect improvement at all, this delta must
come back strongly positive.

- Corpus: 11 rows (12 eval tasks; one task has no usable oracle), maximal
  memorization settings (LoRA r32, lr 2e-4, 12 epochs).
- Expected: delta +0.3 to +0.6 with a CI lower bound above zero.
- Reading: **delta ≫ 0** → instrument works, every negative above is real
  science. **delta ≈ 0** → instrument broken, all prior negatives
  uninterpretable. **delta > 0, CI crosses zero** → underpowered; widen the
  eval before publishing verdicts.

<!-- LEAK-RESULT: pending — replace this block with the measured delta/CI/verdict -->
Status: training complete; paired eval in flight. This document will be
updated with the measured delta.

## Staged plan (each batch gates the next)

| batch | question | cost |
|---|---|---|
| 0 — leak control | does the instrument detect a known positive? | ~$10 |
| 1 — re-run the negatives on a denser eval (38 tasks, 0–1 reward, 3 replicates, pooled baseline) | were the negatives real or eval noise? | ~$30 |
| 2 — step-count ladder (8/14/28/39 calls/row, same data otherwise) | does longer-than-base rollout length flip the delta? | ~$30 |
| 3 — GRPO with turn-level credit assignment | does RL flip corpora that hurt under SFT? | ~$1,150 |

Batches 0–2 together cost ~$70 and buy the right to place the batch-3 bet.

## Relation to OpenEnv

The OpenEnv protocol adapter in this repository serves these same task
packages over OpenEnv's client/server interface (see the compatibility notes
in the README and `docs/native-dataset-openenv-smoke.md`). The end-to-end
OpenEnv smoke evidence to date is a *systems* claim — healthy scored rollouts,
zero infra errors, no quality lift measured through that path yet. This ledger
is the measurement-side counterpart: what a quality claim has to look like
(paired, CI'd, screened-then-confirmed, with a known-positive control) before
it is a claim at all. It is also the shape of evidence the proposed
`validate-task` lane in upstream OpenEnv (huggingface/OpenEnv#898) would
certify for externally contributed task packages.
