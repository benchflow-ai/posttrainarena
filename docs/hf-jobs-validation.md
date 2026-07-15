# HF handoff validation report

## Verdict

The PostTrain Arena Hugging Face handoff is implemented and its exact UV runner
completed the historical pipeline on an H100. July 11 allocation attempts
through the HF Jobs scheduler were blocked by account credits, not code;
current paid-launch availability was not re-tested during the July 15 audit.

This July 11 evidence predates the OpenCode evaluation migration. It validates
the HF bundle, trainer, publishing, and earlier TRL evaluation path. The current
OpenCode evaluator has separate real SkillsBench + Daytona canary evidence, but
the full HF H100 flow must be rerun against the new shared vLLM endpoint and
OpenCode custom-rollout GRPO path.

## Validated flow

Run: `hf-wrapper-h100-fixed-20260711T010948Z`

1. A real `submission.yaml` entry was structurally validated.
2. Its task corpus was uploaded and pinned as
   `benchflow/posttrainarena-submission-smoke@aa2bb9b6f57288eeeffae7e9a39e1b27aa4c8284`.
3. The portable job bundle was uploaded to the Hub.
4. The PEP 723 UV script downloaded that bundle and installed PostTrain Arena
   commit `45fcbaca86f065eed89aee1e949a64462073383d`.
5. Qwen3-4B completed baseline evaluation, reward-1 teacher collection,
   one-step SFT, forced zero-reward GRPO, and final evaluation.
6. The same base/final checkpoint pair was evaluated on:
   - Data Agent:
     `benchflow/data_agent_rl_environment_eval@0ea976c79e3248c85737c4f7363484e4d47ce287`
   - SkillsBench:
     `benchflow/skillsbench@be2a6ce2cb1f4ff67ce937307cade0c5a0477a13`
7. Run artifacts, model weights, job logs, status, and benchmark scores were
   published to the Hub and consumed by the live leaderboard Space.

## Results

- Healthy rollout artifacts: `11/11`
- Runtime/verifier errors: `0`
- Teacher reward: `1.0`
- Primary baseline/final/delta: `0.0 / 0.0 / 0.0`
- Data Agent baseline/final/delta: `0.0 / 0.0 / 0.0`
- SkillsBench baseline/final/delta: `0.0 / 0.0 / 0.0`
- Macro benchmark delta: `0.0`
- Forced GRPO ran: `true`
- Final BF16 model size: `8,044,982,080` bytes

Performance change was explicitly out of scope; this run validates execution
and artifact integrity.

## Published evidence

- Artifacts:
  `https://huggingface.co/datasets/benchflow/posttrainarena-hf-jobs-smoke-results/tree/main/runs/hf-wrapper-h100-fixed-20260711T010948Z`
- Artifact commit: `72f37c7692f1af5b72cc5e2c94d1f5e7c2c0dc34`
- Model:
  `https://huggingface.co/benchflow/posttrainarena-hf-job-qwen3-4b/commit/d381be6366c7487d8f2d3b6fa4df37067c99f953`
- Leaderboard commit:
  `https://huggingface.co/datasets/benchflow/posttrainarena-leaderboard/commit/7163e258a468984df89dee18b4fca822dc94e45f`
- Live Space:
  `https://huggingface.co/spaces/benchflow/posttrainarena-leaderboard`
- Pipeline W&B:
  `https://wandb.ai/benchflow-ai/posttrainarena-hf-jobs/runs/2mvndq4g`
- Benchmark W&B:
  `https://wandb.ai/benchflow-ai/huggingface/runs/mbyfkxwy`

## HF Jobs scheduler blocker

Real `hf jobs uv run` requests reached the official API under the `benchflow`,
`bingran-you`, and `xdotli` namespaces. Each returned HTTP `402 Payment
Required` with the same reason: insufficient prepaid Jobs credits.

The uploaded script and bundle were therefore executed on a dedicated H100
outside the HF scheduler. Authenticated job listing succeeded on July 15 and
showed no active jobs, but that read-only check does not prove credits are now
available. The next paid scheduler retry must validate the current
Qwen3.5/OpenCode path, including authenticated model ingress, Docker
availability, and separate trainer/vLLM GPU placement; those constraints may
require topology or secret-boundary changes.

## Cleanup

- H100 instance `89a645f8fb074dce840099871aee1a15` terminated.
- Remote credential file deleted before termination.
- GPU utilization and allocated memory returned to zero.
- Daytona listed zero sandboxes created during the final run window.
