# Hugging Face training status

Verified September 21, 2026. This page describes the separate Submission Lab, not a replacement for the checked-in public OpenCode/TRL pipeline.

## What works

- Hugging Face GPU Jobs completed Qwen3.6-27B NF4 LoRA training on A100 Large hardware. The general run performed 20 optimizer steps and saved an adapter; a separate reload/inference smoke checked the saved adapter.
- A separate verifier-guided SFT checkpoint search ran on one deliberately seen Java repair task. The baseline passed 0/3 checks. The 4- and 8-step candidates remained at 0/3 and were rolled back; the 16-step candidate passed 3/3 and was retained.
- The search performed 28 optimizer steps in total and retained 16. The accepted in-memory checkpoint passed the verifier; a separate reload-and-verifier test of that particular hillclimb adapter has not been established.
- At the audit, no training jobs were running. The Space being online is not an active training job.

The three checks cover note existence, patch existence, and Java build success on one task. This is seen-task overfitting, not three independent tasks, GRPO, a 41-task benchmark result, or held-out generalization.

## Access and remaining work

The lab, jobs, and artifacts require explicit access. Private Hugging Face resources can return 404 to an unauthorized reader; signing in alone does not grant access. The [website cookbook](https://posttrain.com/docs/cookbook) provides screenshots and read-only inspection examples for authorized users.

The current lab uses a durable single-run reservation and blocks repeat GPU submissions even after completion. Hillclimb is a separate runner, not a selectable mode in the normal form. A portable fresh-run launcher, preflight, and per-run budget accounting remain needed for self-service reproduction.

## Public pipeline boundary

The repository still contains the Qwen3.5-9B OpenCode/TRL recipes described in [the training guide](training-pipeline.md). Those files have not been converted to the lab's Qwen3.6-27B runner. The newer single-GPU HF result does not validate the older recipe's Docker, ingress, and two-physical-GPU topology on HF Jobs.

The July HF credit failure is historical and no longer describes the project's overall ability to execute HF GPU jobs. Keep the [July validation report](hf-jobs-validation.md) as dated evidence for that runner, not as a current account-status check.
