# Recipe v2: Qwen3.5-35B-A3B, whole-collection GRPO, one 8×H200 node

Status (Sept 24, 2026): a planning template, **not run**. No number below has been measured on this model, node or recipe. Template: `pipelines/benchflow-task-posttrain/configs/qwen3.5-35b-a3b-terminal-grpo-v2.toml`. The tables from `[runtime]` down are the method; the Space composer supplies model, datasets, suites and task lists.

## What changes from grpo-v1

| | grpo-v1 (tb2-9b smoke) | recipe v2 |
|---|---|---|
| Model | Qwen3.5-9B | Qwen/Qwen3.5-35B-A3B @ `59d61f3` (35B MoE, 3B active) |
| Training data reached | one task (1 group of 8, TRL shuffle) | every accepted task, seeded `cover` schedule, logged per step |
| Rollouts per run | 8 generated, 2 used in a gradient step | 32 steps × 8 tasks × 8 = 2,048, all used |
| Optimizer step | one rollout (`gradient_accumulation_steps = 1`) | one full generation batch of 64 rollouts |
| LoRA learning rate | 1e-6 | 1e-5 |
| Failed rollouts | stop training | masked from loss and baseline, counted |
| Held-out | TB2 32-task subset, 1 trial | TB2 86 + LHTB 38, 3 trials each, per-suite and pooled Δ with SE |
| Records | `score.json` | plus `score_v2`, `train_task_stats.json`, `train_sampler.json`, `provenance.json` |

## Choices and evidence

**Steps and batch: 32 steps of 8 tasks × 8 rollouts.** The budget is set by wall-clock, not by what the literature says is enough. Published runs that moved held-out terminal or SWE scores used 10-60 times more rollouts: TMax 500 steps × 8 prompts × 32 (about 128K rollouts; arXiv 2606.23321 §4.1 and Table 13); LEGO-RL on this exact model 126 steps × 64 prompts × 8 (about 64.5K; arXiv 2608.17393 Table 10); the open MiMo-V2.6 9B recipe 200 steps × 32 × 16 (about 102K; `XiaomiMiMo/verl` `recipes/code/config/train.yaml`); SETA 8 tasks × 16 trajectories per step on one 8×H200 node (arXiv 2607.10891 Tables 8-9). SETA's step count appears only in its training-curve figure (Fig. 5); the "about 150 steps" used in earlier PTA notes comes from that plot and is not stated in the text. If it is right, SETA is about 19K rollouts. SETA moved Qwen3-8B on TB2 from 3.1 ± 0.6 to 10.7 ± 1.3 (Table 3). 2,048 rollouts is 128 times grpo-v1, but it is small by these standards. **Whether it moves TB2 or LHTB at all is unknown.** It is the largest budget I estimate fits about 2 node-days (below). With `require_full_coverage`, 32 × 8 = 256 group slots means collections of up to 256 tasks are each sampled at least once. A larger collection needs more steps, or the run refuses to start. Under a fixed step budget, Δ values a collection per step, not in total (attribution report §3.5).

**Group size 8.** Group sizes in the range: LEGO-RL 8 (same model, OpenCode), Nebius 10 (arXiv 2508.03501 §4.3), SETA and MiMo 16 (MiMo report §4.1: 1,568 prompts × G = 16), TMax 32. TMax found 32 more stable than 8 over 500 DPPO steps on Qwen3.5-9B (§5.2, Fig. 7); its runs often collapsed past about 300 steps (§5.2), far beyond 32 steps. At a fixed rollout budget, G = 8 reaches twice as many tasks per step, which per-task attribution needs. The cost is more groups with no reward spread: all 8 rollouts fail with probability (1 − p)^8, for example 17% at p = 0.2 (2.8% at G = 16). LEGO-RL measured 43% zero-variance groups under OpenCode. If training is unstable or most groups carry no signal, switch to 16 × 4 tasks (same rollouts per step, half the coverage). The pipeline does not resample zero-variance groups the way DAPO dynamic sampling, SETA, TMax and MiMo do. Screening submissions to tasks the base model sometimes solves (the proposed 1-7-of-8 band gate) is the upstream fix; LEGO-RL's unscreened pool "improves on neither measure" (§4.3).

**LoRA, not full fine-tuning; learning rate 1e-5.** None of the terminal/SWE RL papers above uses LoRA. Nebius states it updates all parameters, and MiMo's open recipe is full fine-tuning. Their learning rate is 1e-6 (TMax, LEGO-RL, Nebius, MiMo; SETA uses 1.7e-5 with an FSDP trainer). This pipeline's trainer is a single process that splits the model layer by layer over GPUs 0-3 (TRL's default `device_map="auto"`), with no FSDP or ZeRO. Full fine-tuning of 35B parameters needs about 16 bytes per parameter for bf16 weights, gradients and fp32 Adam states: about 560 GB, the entire 564 GB of four H200s before activations. It does not fit without a different trainer. LoRA keeps the frozen bf16 weights (about 70 GB) plus small adapters. Schulman and Thinking Machines, "LoRA Without Regret" (Sept 29, 2025), report that LoRA "fully matches the learning performance of FullFT when running policy gradient algorithms for reinforcement learning, even with ranks as low as 1". They also report that the best LoRA learning rate "is consistently 10x the one used for FullFT" (about 15x for short runs), hence 1e-5; grpo-v1's 1e-6 is likely about 10 times too low. The same source warns that "attention-only LoRA significantly underperforms" and that MLP and MoE layers matter most. **This is the largest open risk of the template:** in transformers 5.13, Qwen3.5-MoE stores its 256 routed experts as fused 3-D `nn.Parameter` tensors, which PEFT's `target_modules = "all-linear"` skips. As written, the template adapts attention, linear-attention, shared-expert and output projections, but not the routed experts. `grpo.lora_target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]` (PEFT `target_parameters`) targets them. It is commented out until a validation run confirms three things: PEFT wraps these tensors, TRL's merge-and-sync to vLLM carries them (the policy-attestation check compares logprobs), and memory holds. The Thinking Machines MoE experiments used a separate LoRA per expert with rank = total rank / active experts; how PEFT's `target_parameters` splits rank over experts is unverified.

**Context: 64K trainer budget.** The Sept 24 context grid (Qwen3.5-9B, sealed `tb2-32`, one A100 serving plain vLLM, concurrency 8, 1,800 s limit) found no context failures at 64K or 128K: 128K with OpenCode scored 4/32, 64K with the lighter pi-acp harness 2/32, and a canceled 64K OpenCode point 0/10, against 2–3/32 for the challenge path at 32K behind the bridge. Half of the 128K tasks and 26 of 32 pi-acp tasks hit the time limit instead: eight agents shared about 30–76 generated tokens per second. Only 32K without the bridge failed on context (8 of 8 on the first prompt). So `max_completion_length = 65536` stays; 128K would double the output-head logits for no measured gain, and the next lever is serving throughput, not context. Reference points: TMax uses a 65,536-token response budget for Qwen3.5 (Table 13); Nebius trained at 65K and then 131K (§4.3); LEGO-RL uses 200K (30K prompt + 170K response) on this model; SETA 28,672. Against a plain vLLM server at `max_model_len` 32,768, OpenCode failed 8 of 8 tasks on the first prompt (grid point g1). Two constraints are specific to this pipeline:
- **Trainer memory.** TRL 1.8 materializes full-vocabulary logits for every completion token: tokens × 248,320 (the Qwen3.5-MoE vocabulary in transformers' default config; I assume the checkpoint matches) × 2 bytes ≈ 33 GB in bf16 at 64K, before the fp32 log-softmax and the backward pass, all on the GPU that holds the output head. 64K is borderline on a 141 GB H200; 128K is not expected to fit without a chunked or fused loss (TRL's Liger path, not a dependency today). Under `rollout_failure_policy = "mask"`, longer trajectories are truncated to the budget and keep their reward.
- **Train/eval context mismatch.** The Space job script caps the bridge's logprob path at 16,384 tokens (`--max-logprob-context-tokens`), while evaluation uses the full context. Training rollouts therefore see a shorter context than the held-out runs. The job script must raise both caps to the chosen context.

**Timeouts: 1,800 s agent wall and idle, 600 s sandbox setup.** The 64K and 128K grid points use 1,800 s. The measured 9B challenge run with 900 s had 26 of 64 rollouts hit the timeout (`challenge-5503c8d51aa9`). Idle equals wall because the pipeline treats `idle_timeout` rows as infrastructure errors; a scored wall-clock timeout is a failure, as in Terminal-Bench. One timeout applies to every task and to both training and evaluation. That overrides TB2's per-task timeouts, and LHTB's intended task durations are not known to me: if LHTB tasks need more than 30 minutes, they will time out as failures. LEGO-RL uses stage-specific timeouts and masks wall-clock timeouts from training; PTA scores them. Which of the two is right for training is an open choice.

**Infrastructure errors are masked, not scored 0.** `rollout_failure_policy = "mask"`: a rollout with no healthy verifier-scored result after two attempts gets reward `None` and no trainable tokens. TRL 1.8 leaves it out of the group mean and std and gives it zero advantage. It is counted per task in `train_task_stats.json`. LEGO-RL excludes infrastructure failures "from group-relative advantage estimation and the policy loss" (6.4% of OpenCode trajectories, §4.3). The MiMo-V2.6 penalty module keeps "infrastructure failures out of the training signal", and "a sample with no surviving sequence is rejected" (report §6.1). Over-length trajectories are truncated, not dropped, because Nebius found that discarding them removes negative examples of looping (§5.2). Training stops if a whole batch fails or more than 25% of rollouts are masked, which is about 4 times LEGO-RL's OpenCode rate.

**Held-out: TB2 86 + LHTB 38, 3 trials each.** Three trials follow TMax (each evaluation run 3 times, §4.1); SETA used 8 runs and Nebius 10. The attribution check estimates the SE of Δ at 1.6-2.3 pp for about 88 TB2 tasks × 3 trials against a pooled 10-trial baseline, from evaluation noise only. It rises to about 3 pp with one training seed at a seed SD of 2 pp; the minimum detectable effect is 2.8 × SE. With an in-run 3-trial baseline the SE is somewhat higher. My rough scaling, not measured: LHTB's 38 tasks give about √(86/38) ≈ 1.5 times TB2's SE. `score_v2` reports each suite and the task-weighted pool, with the formula in `docs/training-pipeline.md`. Trials are only informative if sampling is stochastic; the OpenCode/bridge sampling temperature has not been checked.

**MoE-specific stability.** For this model, LEGO-RL found that replaying rollout-time expert routing in the trainer raises rollout-training probability correlation from 0.9946 to 0.9993 (§4.4). TRL has no routing replay. TMax adds an FP32 output head for Qwen3.5 because the hybrid architecture produces logprob mismatch spikes (Fig. 3); this pipeline has none. It relies on TRL's vLLM importance-sampling correction, whose default mode masks sequences with out-of-range ratios. Watch `sampling/sampling_logp_difference` and `sampling/importance_sampling_ratio` in the training log.

## Wall-clock estimate and fit

**GPU fit.** LoRA fits one node on paper: the trainer holds about 70 GB of frozen weights over GPUs 0-3, and vLLM serves the policy on GPUs 4-7. The binding memory is the 64K logits on the output-head GPU (above). The vLLM layout for a 35B MoE (tensor parallel versus replicas on GPUs 4-7) is a job-script decision. Tensor parallelism was a dead end on HF A100 nodes without peer-to-peer links; H200 nodes normally have NVLink, which is unverified here. Full fine-tuning does not fit this trainer.

**Time.** The model is: step ≈ rollout makespan + trainer compute + weight sync, where rollout makespan ≈ (R × d̄ / C) + straggler tail.
- Inputs: R = 64 rollouts per step; C = 32 concurrent sandboxes (assumed; see unknowns). d̄ = mean rollout duration including sandbox setup. The only measured anchor is the 9B challenge run at a 900 s timeout: mean 760 s, maximum 1,257 s. Scaled to a 1,800 s timeout, I assume d̄ ≈ 22 min and a maximum of about 36 min.
- Trainer compute: about 64 × 40K ≈ 2.6M tokens per step at roughly 45 GFLOP per token. That covers 3B active parameters over forward, recompute, backward and the old-logprob pass, plus 10 full-attention layers. At an effective 100-300 TFLOP/s on one active GPU, that is 7-20 minutes. Weight sync (merge LoRA, broadcast about 70 GB) adds a few minutes. Unmeasured.

| Phase | Estimate | Basis |
|---|---|---|
| Setup: model download and load, snapshots, base sync, gate (8 tasks) | 1-1.5 h | 35B weights; one gate job |
| Baseline: 86 × 3 TB2 + 38 × 3 LHTB, 6 bench jobs | about 7 h (5-9) | 372 cells × 22 min / 32, plus a tail per job |
| Training: 32 steps × (55-80 min rollouts + 10-20 min trainer) | about 40 h (35-53) | model above |
| Merge, save, sync; final evaluation (same as baseline) | about 7.5 h (5.5-9.5) | |
| **Total** | **about 56 h ≈ 2.3 node-days (1.9-3.0)** | |

This does **not** fit the 1-2 node-days per run assumed in the attribution check's budget (§4.1). At 2.3 node-days, the 396 node-days of Oct 5-Nov 7 hold about 170 runs if nothing else runs. Levers, largest first:
- Measure the baseline once per pinned stack with 10 or more trials (the attribution check's recommendation), and skip the in-run baseline. That saves about 7 h per run, but needs a pipeline option this branch does not add.
- Raise sandbox concurrency to 64. The step then approaches the single-rollout ceiling of about 36 min plus trainer time.
- Use a shorter training timeout than evaluation timeout. Today one harness setting drives both.
- Run all trials of a suite in one bench job, which saves straggler tails.
- Asynchronous rollouts. LEGO-RL measured about 1.0 h per step asynchronously versus 1.9 h synchronously after compute correction (§4.6). That is a trainer redesign.

Conversely, by the same model, the literature-scale budget of about 150 steps of 64 rollouts would take about 7-11 node-days per run with this synchronous pipeline.

## Unknowns, stated plainly

- Nothing here has run. The template's numbers are estimates from one 9B run and paper figures.
- Whether 2,048 rollouts of LoRA moves TB2 or LHTB for this model at all.
- Whether expert LoRA (`lora_target_parameters`) works end to end through PEFT, TRL's merge-and-sync and vLLM; and how much all-linear-only LoRA loses without it.
- Trainer memory at 64K (and certainly at 128K) on the output-head GPU; trainer throughput for a 35B MoE under `device_map="auto"`.
- Daytona capacity for 32 concurrent sandboxes. On Sept 24 the shared org had about 210 vCPU, about 195 of them in use by other teams. Whether Nebius runs come with their own sandbox capacity is also unknown.
- Rollout durations and pass rates of the 35B model under OpenCode at 1,800 s, and LHTB's intended timeouts.
- Whether thinking mode fits the 64K budget, and rollout speed per agent at concurrency 32 on the H200 layout (the grid's time-outs came from slow shared serving). SETA switched Qwen3-8B to non-thinking mode because thinking overflowed its response limit.
- vLLM serving layout and support for this checkpoint under the pinned vLLM ≤ 0.23. The tool-call and reasoning parsers for Qwen3.5 must be set as for the 9B.
- Whether eval sampling is stochastic enough for trials to be independent.

## Before the first challenge run

1. One-step smoke on the node (about 2-3 h): `max_steps = 1`, `trials = 1`, a 16-task collection, a small eval subset. Check trainer memory at the chosen context, weight sync plus attestation, masking, and every report file. Record rollout durations and the step time.
2. The same run with `lora_target_parameters` enabled. Keep it only if attestation passes and memory holds.
3. Replace the estimates here with the measured step and eval times, then fix `max_steps` and concurrency.
4. Run the attribution check's known-answer tests (an A/A run with two seeds, and a planted positive control) before ranking anything.

## Sources

- TMax, arXiv 2606.23321: §4.1 recipe, Appendix D.1 Table 13 hyperparameters, Table 3 and Table 16 results, §5.2 and Figs. 3, 6, 7 stability.
- SETA, arXiv 2607.10891: §4.1 setup, Table 3 results, Appendix A.8 Tables 8-9 hyperparameters, Fig. 5 training curves.
- LEGO-RL, arXiv 2608.17393: §4.2-4.6 and Table 10 (Qwen3.5-35B-A3B under OpenHands SDK, Claude Code and OpenCode).
- Nebius, Golubev et al., arXiv 2508.03501: §4.3 recipe, §5.2 findings, Appendix C-D.
- MiMo-V2.6 technical report (HF `XiaomiMiMo/MiMo-V2.6-Pro-RL`): §4.1 batch, §6.1 penalty module; open recipe `XiaomiMiMo/verl` `recipes/code/config/train.yaml`.
- Schulman and Thinking Machines Lab, "LoRA Without Regret", Sept 29, 2025, https://thinkingmachines.ai/blog/lora/.
- Miller, "Adding Error Bars to Evals", arXiv 2411.00640 (held-out standard error).
- PTA attribution check of Sept 24, 2026 (§3.2 power table, §4.1 compute envelope, §4.3).
