# Qwen3.5 Data Agent SFT-to-GRPO validation

This document retains the historical July 14 soccer canary and the clean
July 15, 2026 lift run that supersedes its GRPO claim boundary.

## Clean exact-ID lift run

The run `qwen35-9b-redwine-full-v3-main-69e37ed7` used 16 red-wine training
tasks and 14 disjoint held-out tasks on two H100 80 GB GPUs. OpenCode was the
agent harness for teacher collection, baseline/SFT/final evaluation, and every
GRPO rollout.

The path completed with:

- strict `16/16` verifier-approved Qwen3.5-397B-A17B teacher coverage
- 63 validated tool-calling TRL SFT rows
- one bf16 LoRA SFT epoch, loss `0.142529`
- 128 OpenCode GRPO rollouts: 16 tasks × 8 generations
- four mixed-reward groups and 30 nonzero-gradient optimizer steps
- finite GRPO loss `-0.000946`
- all 248 LoRA-B tensors updated, with no non-finite tensors
- 14/14 healthy paired baseline/final evaluation artifacts

| Stage | Held-out pass rate |
| --- | ---: |
| Baseline | `8/14` (`57.1%`) |
| After SFT | `8/14` (`57.1%`) |
| After SFT + GRPO | `11/14` (`78.6%`) |
| Paired delta | `+3/14` (`+21.4` percentage points) |

The final pass set was a strict superset of the baseline pass set: tasks
`0047_164_47164651_qa_1`, `0060_546_60546361_qa_2`, and
`0095_395_95395894_qa_2` improved, with zero regressions. The paired bootstrap
95% interval for pass-rate lift was `[0.0%, 42.9%]`; this is real single-run
evidence on the 14-task canary, not a competition-scale statistical claim.

PostTrainArena commit
`cf824b214e5ae08d6fc21becbcba7aae55e5109e` produced the run. The runtime used
BenchFlow commit `6d6d2ee0965bdc7fe1e38555d1f7c4c21ee8a840`, whose OpenCode
`1.17.20` pin was merged from BenchFlow PR #931 as
`2a97db55947d6742b765ad34ddd91d74c20d625f`.

## Historical soccer canary

On July 14, 2026, the earlier Qwen3.5 organizer pipeline completed a real
eight-training-task/three-held-out-task run on two H100 80 GB GPUs with Docker
and OpenCode.

## Pinned inputs

- Student: `Qwen/Qwen3.5-9B`
  (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`)
- Teacher route: `openrouter/qwen/qwen3.5-397b-a17b`
- Declared teacher source: `Qwen/Qwen3.5-397B-A17B`
  (`8472618112abcbd45acbcdc58436aff4233c23f7`)
- Train dataset:
  `benchflow/data_agent_rl_environment_train@34ff63c91731df6b3670bfcd7e3d44e6790ddc48`
- Eval dataset:
  `benchflow/data_agent_rl_environment_eval@0ea976c79e3248c85737c4f7363484e4d47ce287`
- Domain: `hugomathien/soccer`
- Run: `qwen35-9b-soccer-20260714T000539Z`

The checked-in canary attempts eight training tasks for up to three teacher
rounds and requires at least four verifier-approved trajectories. The full
organizer recipe remains stricter: it requires one verified trajectory for
every selected training task.

The run was resumed while this canary threshold was being debugged: its first
teacher manifest still declared strict `8/8` coverage, while the effective
continued recipe accepted the four verified trajectories. The same branch now
validates the exact task IDs, teacher provenance, threshold, and coverage mode
before reusing teacher state, so this kind of recipe drift fails closed. A
clean run from the final checked-in recipe remains part of the next
matched-domain quality experiment.

## Completed path

1. The served base model ran three held-out tasks through OpenCode.
2. The Qwen3.5-397B-A17B teacher attempted all eight training tasks and produced
   four selected reward-`1.0`, tool-bearing trajectories.
3. BenchFlow converted those trajectories into 30 validated TRL
   prompt/completion/tools rows.
4. Qwen3.5-9B completed one bf16 LoRA SFT epoch and wrote both the adapter and
   merged checkpoint.
5. The SFT checkpoint synchronized into the shared vLLM endpoint.
6. OpenCode evaluated the SFT model on the three held-out tasks and all eight
   training tasks.
7. TRL collected 16 OpenCode GRPO rollouts with sampled token IDs, logprobs,
   action masks, and BenchFlow verifier rewards.
8. One LoRA GRPO epoch completed and wrote the GRPO adapter and merged
   checkpoint.
9. The final checkpoint synchronized and ran the held-out evaluation through
   OpenCode.
10. The pipeline wrote paired lift and `score.json` reports.

## Metrics

| Stage | Pass rate / reward |
| --- | ---: |
| Held-out baseline | `1/3` (`33.3%`) |
| Held-out after SFT | `1/3` (`33.3%`) |
| Post-SFT training gate | `4/8` (`50.0%`) |
| Held-out after GRPO | `1/3` (`33.3%`) |
| Held-out delta | `0.0` |
| Training tasks after GRPO | `3/8` (`37.5%`) |

SFT trained on 30 rows for one epoch with aggregate train loss `0.147735`.
GRPO completed one epoch over 16 rollouts with rewards:

```text
[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
```

Aggregate GRPO train loss was `0.001294`. The SFT and GRPO adapter files have
different SHA-256 digests. A later exact-token audit found that the old rollout
parser independently reconstructed prompts and matched the serving endpoint's
prompt-token count on `0/321` sampled agent exchanges. The reported GRPO loss
and adapter change therefore do not establish a valid policy update.

## Runtime fixes exercised or derived from the canary

- The GPU bootstrap selects vLLM's official `0.23.0+cu129` wheel instead of the
  CUDA 13 PyPI wheel on CUDA 12.x H100 hosts.
- Scored zero-tool completions remain legitimate model failures rather than
  being misclassified as broken evaluation artifacts.
- `posttrainarena-vllm-serve` maps Transformers Qwen3.5 text-policy parameter
  names onto vLLM's official multimodal wrapper during TRL weight
  synchronization.
- The model bridge parses Qwen3.5 native
  `<function=...><parameter=...>` tool calls as OpenAI-compatible tool calls.
- The corrected GRPO path consumes exact prompt and completion IDs from the
  bridge sidecar, while still normalizing OpenAI stringified tool arguments for
  conversational history.
- Evaluation and GRPO materialization retain healthy scored retries instead of
  rejecting the whole task because an earlier attempt failed.
- The historical canary used two-generation prompt pairs. The production recipe
  now uses eight generations per task, expandable CUDA segments, and per-step
  cache clearing so long trajectories fit on the 80 GB trainer GPU.
- The final Qwen3.5 recipes synchronize the pinned base checkpoint before
  baseline evaluation so a reused vLLM server cannot contaminate the reference.
- Deterministic policy attestation compares the direct TRL server with the
  public OpenCode bridge after every explicit checkpoint synchronization.
- Resume now rejects any recipe-incompatible persisted run plan and validates
  the exact task IDs and coverage in reused teacher manifests.

## Claim boundary

The historical soccer run proves orchestration only. Its teacher stage crossed
a recipe change during resume and its old rollout parser reconstructed prompt
IDs incorrectly, so its `1/3 -> 1/3` result is not valid GRPO-learning
evidence.

The clean red-wine run above closes that gap: exact served prompt IDs,
provider-sampled logprobs, OpenCode-only rollouts, finite nonzero LoRA updates,
and healthy held-out evaluation jointly establish a valid `8/14 -> 11/14`
post-training lift on the canary slice.
