# Native data-agent OpenEnv smoke

On July 10, 2026, PostTrain Arena completed a real one-train/one-held-out
end-to-end run against the public BenchFlow-native data-agent datasets.

This historical run used the earlier TRL/OpenEnv evaluation loop. It does not
validate the newer OpenCode baseline/gate/final evaluator or the still-pending
OpenCode GRPO endpoint-resynchronization path.

## Pinned inputs

- PostTrain Arena: `fbdd7c2`
- BenchFlow: `0b41232cf02e9c4f22c01e284724dd2a02c3f468`
- OpenEnv: `6823135a714814e3efb3e39c4a9edff01e1a2a98`
- Train dataset:
  `benchflow/data_agent_rl_environment_train@34ff63c91731df6b3670bfcd7e3d44e6790ddc48`
- Eval dataset:
  `benchflow/data_agent_rl_environment_eval@0ea976c79e3248c85737c4f7363484e4d47ce287`
- Runtime: local OpenEnv server/client adapter over BenchFlow and Daytona
- Model: `Qwen/Qwen3-4B`

## Completed stages

The single pipeline invocation completed:

1. native train and eval snapshots;
2. OpenEnv baseline held-out evaluation;
3. reward-`1.0` teacher trajectory collection;
4. tool-aware SFT conversion and validation;
5. one-step LoRA SFT and BF16 weight merge;
6. post-SFT held-out and training-gate evaluation;
7. one forced GRPO step with two rollouts;
8. BF16 GRPO checkpoint save;
9. final held-out evaluation and paired lift report.

All seven scored rollout artifacts were healthy, with no infrastructure or
verifier errors. Scores remained `0.0 -> 0.0`; this validates the system path,
not model-quality lift.

## Artifacts

- Trajectories:
  `benchflow/env0-experiment-trajectories@ba11080731be1f15b17962cbb789972795164ebc`
- SFT checkpoint PR: `benchflow/benchflow-qwen3-4b` discussion `#4`
- GRPO checkpoint PR: `benchflow/benchflow-qwen3-4b` discussion `#5`
- W&B run:
  `benchflow-ai/posttrainarena-native-dataset-e2e/cf7o84cb`

The checkpoint PRs remain open and unmerged.

## Cleanup

Every Daytona sandbox ID referenced by the run was absent during teardown. The
remote secret file was removed, and the dedicated Lambda H100 instance was
confirmed terminated.

The run emitted the known non-fatal asynchronous Daytona cleanup warning after
score artifacts were written. No run sandbox remained active.
