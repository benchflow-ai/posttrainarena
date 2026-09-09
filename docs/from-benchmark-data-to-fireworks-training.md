# From Benchmark Data to Fireworks Training with PostTrainArena

<!-- markdownlint-disable MD013 -->

PostTrainArena connects benchmark datasets to Fireworks by turning successful
agent runs into verified chat training data. We use SkillsBench in this
tutorial, but the same workflow applies to any dataset with tasks and
verifiable outcomes.

## Worked example: overfitting Qwen3.6-27B on SkillsBench

In this tutorial, we use a deliberately contaminated memorization test:

```text
DeepSeek V4 Pro solves 41 SkillsBench tasks without skills
    → retain one verified reward-1 trajectory per task
    → LoRA SFT Qwen3.6-27B on those 41 trajectories
    → evaluate base and tuned Qwen on the same 41 tasks at pass@10
```

This is an **overfitting** experiment, not a held-out benchmark evaluation. A
positive delta would show that the training and evaluation path can make Qwen
reproduce behavior present in its training corpus, but it would not demonstrate
any generalization to novel SkillsBench tasks.

## 1. Add your tasks

Store each task with its instructions, environment, and verifier:

```text
skillsbench/tasks/
  task-name/
    task.md
    environment/
    verifier/
```

The verifier should return a reward so failed runs can be excluded from
training.

The worked example starts from the 87 active tasks under `skillsbench/tasks/`.
Its final 41-task roster is recorded in
`experiments/skillsbench-fireworks-rft/skillsbench_rft/roster.py`.

## 2. Generate trajectories

Run a teacher model on the tasks with BenchFlow and Daytona:

```bash
bench eval run \
  --tasks-dir skillsbench/tasks \
  --agent openhands \
  --model openai/deepseek-v4-pro \
  --sandbox daytona \
  --skill-mode no-skill \
  --jobs-dir jobs/skillsbench-teacher
```

PostTrainArena retains only complete trajectories from the intended model that
receive reward `1.0` and contain no agent, verifier, or skill-use errors.

### DeepSeek V4 Pro no-skills performance

The canonical no-skills run on SkillsBench is available on Hugging Face under:

```text
hf-traces/skillsbench-leaderboard/submissions/skillsbench/v1.1/
  openhands-no-skills__deepseek-v4-pro/
  2026-06-08__pr2-pr3-selected-3trial/
```

Downloading these trajectories lets us build a training corpus with
DeepSeek V4 Pro as the teacher for Qwen3.6-27B.

Build the exact selection manifest with:

```bash
python3 scripts/build_verified_skillsbench_manifest.py \
  --tasks-dir skillsbench/tasks \
  --artifacts-root hf-traces/skillsbench-leaderboard \
  --artifacts-root jobs/deepseek-v4-pro-raw-regeneration \
  --artifacts-root jobs/deepseek-v4-pro-trace-expansion \
  --out pipelines/benchflow-task-posttrain/data/deepseek-v4-pro-no-skills-llm-manifest.json \
  --expected-tasks 87 \
  --expected-selected 41 \
  --require-llm-trajectory
```

The checked-in manifest selects 41 tasks from 52 eligible unique passing
traces and ignores 12 duplicate artifacts (runs).

## 3. Convert verified runs

Convert the selected trajectories into trainer-ready chat JSONL:

```bash
bench train convert . \
  --out data/skillsbench-sft.jsonl \
  --format trl-sft \
  --row-mode rollout \
  --min-reward 1 \
  --canonical-selection data/verified-manifest.json \
  --manifest data/skillsbench-sft-stats.json
```

For the Qwen overfitting run, the concrete paths are:

```bash
bench train convert . \
  --out pipelines/benchflow-task-posttrain/data/sb-ds4pro-no-skills-41-trl.jsonl \
  --format trl-sft \
  --row-mode rollout \
  --min-reward 1 \
  --canonical-selection \
    pipelines/benchflow-task-posttrain/data/deepseek-v4-pro-no-skills-llm-manifest.json \
  --manifest \
    pipelines/benchflow-task-posttrain/data/sb-ds4pro-no-skills-41-trl-stats.json
```

The conversion report records 41 rollouts and 1,363 exchanges, with all 41
rows containing tool calls and no skipped rewards, provider errors, terminal
errors, missing trajectories, or invalid rows.

Existing datasets can skip trajectory generation if they already use standard
chat records:

```json
{"messages":[{"role":"user","content":"Complete this task."},{"role":"assistant","content":"Here is the solution."}]}
```

## 4. Train on Fireworks

Export your API key, then upload the JSONL and launch supervised fine-tuning:

```bash
export FIREWORKS_API_KEY="..."

python3 scripts/fireworks_sft.py \
  --data data/skillsbench-sft.jsonl \
  --dataset-id skillsbench-sft \
  --output-model skillsbench-qwen-sft \
  --base-model accounts/fireworks/models/qwen3p6-27b \
  --epochs 10 \
  --lr 1e-4 \
  --lora-rank 32 \
  --max-context-length 32768
```

PostTrainArena creates and validates the Fireworks dataset, then starts the
training job.

The worked example used the same hyperparameters with:

```bash
python3 scripts/fireworks_sft.py \
  --data \
    pipelines/benchflow-task-posttrain/data/sb-ds4pro-no-skills-41-trl.jsonl \
  --dataset-id sb-ds4pro-no-skills-41-trl \
  --output-model sbds41-r32-e10 \
  --base-model accounts/fireworks/models/qwen3p6-27b \
  --epochs 10 \
  --lr 1e-4 \
  --lora-rank 32 \
  --max-context-length 32768
```

## 5. Evaluate the result

For a normal experiment, deploy the trained model on Fireworks and evaluate it
on held-out tasks using the same agent, sandbox, model sampling, timeout, and
retry configuration as the base model.

For this overfitting experiment, evaluate both models on the same 41 training
tasks with ten attempts per task:

```text
base:
  accounts/fireworks/models/qwen3p6-27b
  through deployment probe-b2

tuned:
  accounts/xiangyi-0cb55c/models/sbds41-r32-e10-50cac45a
  through deployment sbds41-pass10

shared:
  agent = opencode
  sandbox = daytona
  skill mode = no-skill
  tasks = the 41 entries in skillsbench_rft/roster.py
  trials = 10
```

Use `experiments/skillsbench-fireworks-rft/scripts/run_pass_at_k.py` for the
41-task evaluation. Point
`experiments/skillsbench-fireworks-rft/eval-matrix.yaml` at one deployment at
a time so only one paid deployment is active.

After each arm finishes, scale its Fireworks deployment to zero and verify the
active replica count is zero. See
[`fireworks-deployment-safety.md`](fireworks-deployment-safety.md).

Report the result as a contaminated-task comparison:

```text
Qwen3.6-27B base
  healthy attempts: 410/410
  pass@10: 19/41
  per-attempt pass rate: 121/410
  mean verifier reward: 0.314

Qwen3.6-27B + sbds41-r32-e10
  healthy attempts: 410/410
  pass@10: 37/41
  per-attempt pass rate: 247/410
  mean verifier reward: 0.631
```

The complete path is:

```text
Tasks → Agent trajectories → Verification → Chat JSONL → Fireworks SFT → Evaluation
```

SkillsBench supplies the benchmark tasks and verifiers. PostTrainArena handles
selection and conversion. Fireworks trains and serves the resulting model.
