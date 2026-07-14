# Qwen3.5 Data Agent SFT-to-GRPO canary

On July 14, 2026, the Qwen3.5 organizer pipeline completed a real
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
- GRPO uses one two-generation prompt pair at a time, expandable CUDA segments,
  and per-step cache clearing so long trajectories fit on the 80 GB trainer GPU.
- The final Qwen3.5 recipes synchronize the pinned base checkpoint before
  baseline evaluation so a reused vLLM server cannot contaminate the reference.
- Deterministic policy attestation compares the direct TRL server with the
  public OpenCode bridge after every explicit checkpoint synchronization.
- Resume now rejects any recipe-incompatible persisted run plan and validates
  the exact task IDs and coverage in reused teacher manifests.

## Claim boundary

This run proves the Qwen3.5 model, teacher, OpenCode harness, native BenchFlow
tasks, LoRA SFT, checkpoint transport, retry handling, and final pass-rate
reporting components execute together. Because the teacher stage crossed a
recipe change during resume and the old GRPO path used mismatched reconstructed
prompt IDs, it is orchestration evidence rather than a valid reproduction of
the final training contract.

It does **not** demonstrate model-quality lift. Held-out score remained
`1/3 -> 1/3`, and the single-run training-task check decreased from `4/8` after
SFT to `3/8` after GRPO. The next quality experiment must use a denser
domain-matched train/eval slice with more verified teacher coverage and the
corrected exact-ID rollout path.
