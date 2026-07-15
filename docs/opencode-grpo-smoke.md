# OpenCode SFT-to-GRPO smoke

<!-- markdownlint-disable MD013 -->

On July 13, 2026, the Qwen3-4B reference pipeline completed a real
OpenCode-only teacher, evaluation, and GRPO run on SkillsBench tasks with
Daytona sandboxes.

This run validates the agent-harness and optimization plumbing. It does not
demonstrate model-quality lift: every student evaluation and both GRPO rewards
were `0.0`.

## Pinned inputs

- PostTrain Arena: PR `#16`
- BenchFlow: `93e58a2bd730a8ff3ca5aff5247aec845a370d1c`
- TRL: `1.8.0`
- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- SkillsBench revision: `be2a6ce2cb1f4ff67ce937307cade0c5a0477a13`
- Training task: `3d-scan-calc`
- Held-out evaluation task: `citation-check`
- Run ID: `live-skillsbench-sft-grpo-20260713T0350Z`

## Runtime topology

- One H100 ran TRL SFT and GRPO.
- A second physical H100 ran `trl vllm-serve`.
- `posttrainarena-train model-bridge` exposed the synchronized student through
  an authenticated OpenAI-compatible endpoint.
- OpenCode inside Daytona used the public bridge URL.
- The trainer used local control URLs for TRL weight synchronization and
  sampled-logprob retrieval.

TRL server-mode synchronization rejected a one-GPU prototype because trainer
and inference roles shared one CUDA device. The validated topology isolates
those roles on distinct physical GPUs.

## Completed stages

1. OpenCode baseline evaluation on the held-out task.
2. OpenCode teacher collection on the training task.
3. BenchFlow reward and artifact-health filtering.
4. Native `trl-sft` conversion and validation.
5. One-step LoRA SFT and merged-checkpoint export.
6. SFT checkpoint synchronization and OpenCode evaluation.
7. OpenCode training-task reward gate.
8. Two OpenCode GRPO rollouts through `GRPOTrainer.rollout_func`.
9. One TRL optimizer step and GRPO checkpoint export.
10. Final checkpoint synchronization and OpenCode held-out evaluation.
11. Paired lift and score-report generation.

The teacher rollout earned reward `1.0`, used 10 tools, and produced nine
validated TRL rows. The SFT step reported loss `0.8664`.

## GRPO artifacts

Both rollouts produced aligned `prompt_ids`, `completion_ids`, sampled
`logprobs`, action masks, and BenchFlow verifier rewards:

| Rollout | Prompt tokens | Completion tokens | Action tokens | Masked context | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 9,994 | 3,610 | 680 | 2,930 | 0.0 |
| 1 | 11,735 | 594 | 594 | 0 | 0.0 |

TRL completed one training step in 142.36 seconds. Because both rewards were
zero, the reported GRPO training loss was `0.0`; this run proves execution and
state synchronization, not a useful policy gradient.

## Evaluation

| Stage | Task | Reward | Agent errors | Missing/malformed LLM trajectory |
| --- | --- | ---: | ---: | ---: |
| Baseline | `citation-check` | 0.0 | 0 | 0 |
| SFT | `citation-check` | 0.0 | 0 | 0 |
| GRPO gate | `3d-scan-calc` | 0.0 | 0 | 0 |
| Final | `citation-check` | 0.0 | 0 | 0 |

The final paired result was `0.0 → 0.0` with delta `0.0`.

## Runtime findings

The live run added coverage for behavior that no-spend tests could not prove:

- OpenCode emits helper calls such as title generation alongside agent calls;
  BenchFlow call-purpose metadata now keeps those out of GRPO action tokens.
- Structured tool calls can be canonicalized on later turns. Reconstruction
  retains exact sampled prefixes, masks rewritten context, and starts a new
  causal segment when OpenCode refreshes dynamic system context.
- Explicit failed provider attempts are excluded when OpenCode records a later
  successful retry.
- The public OpenCode inference URL and trainer-local logprob control URL are
  separate.
- Each model turn is capped at 4,096 generated tokens while the rollout-level
  budget remains independently enforced.
- Verified rollout artifacts are resumable.
- The trainer-owned vLLM communicator is closed after GRPO so final checkpoint
  synchronization can create a fresh communicator.

## Health and claim boundary

The final active artifact tree contained two healthy `grpo_tokens.json` files,
two successful endpoint-sync reports, complete scored BenchFlow trajectories,
and no active rollout errors. A scan of reports, results, jobs, and converted
data found no provider, Daytona, Hugging Face, or bridge secret values.

This is evidence that OpenCode is the sole agent harness across teacher
collection, baseline/SFT/gate/final evaluation, and GRPO rollouts. It is not
evidence of benchmark improvement, broad generalization, or sealed-test lift.
