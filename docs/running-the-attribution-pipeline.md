# Running the attribution pipeline

The attribution pipeline answers, end to end: does *this* model (optionally
trained on *this* data) beat its own base on *this* eval mixture — with a
paired bootstrap confidence interval and a ≤20% per-domain cap? It is the same
instrument behind every number in
[`sft-attribution-ledger.md`](sft-attribution-ledger.md).

## Prerequisites

1. **BenchFlow CLI** (the runner for tasks, sandboxes, and paired lift):

   ```bash
   uv tool install --python 3.12 --prerelease allow benchflow
   ```

2. **A sandbox backend** — one of:
   - `docker`: a local Docker daemon (simplest);
   - `daytona`: cloud sandboxes (`DAYTONA_API_KEY` in the environment);
   - `modal`: cloud sandboxes (a configured Modal profile).

3. **A served model, twice.** Baseline and candidate should be served from the
   same deployment/hardware — e.g. a base model plus a LoRA adapter loaded onto
   the same server — so the delta is the weights and nothing else. Any
   OpenAI-compatible endpoint works.

4. **Tasks.** For the SkillsBench mixture used in the ledger:

   ```bash
   git clone https://github.com/benchflow-ai/skillsbench
   ```

## Run

Copy `configs/attribution.example.yaml`, fill in the model references and
endpoint, then:

```bash
MODEL_API_KEY=... python3 scripts/pta_attribute.py \
  --spec configs/attribution.example.yaml --output report.json
```

`--dry-run` prints the exact `bench eval run` commands without spending
anything. The report contains the capped aggregate delta, per-component paired
lift, and bootstrap CIs. Secrets never live in specs: `${VAR}` references in
`extra_eval_args` expand from the environment at run time and the run exits
loudly on an unset variable.

## Supporting scripts

| script | role |
|---|---|
| `scripts/pta_attribute.py` | the orchestrator described above |
| `scripts/scoring.py` | capped aggregation (paired bootstrap, ≤20% per-domain water-filling) |
| `scripts/build_skillsbench_sft.py` | render (instruction → oracle solution) SFT rows from a tasks dir, with the eval set held out (`--only` inverts the holdout to build a deliberate leak-control corpus) |
| `scripts/fireworks_sft.py` | upload a chat JSONL as a Fireworks dataset and launch the managed LoRA SFT job (the training step in the method above); `--status` polls a job |
| `scripts/train_leak.py` | LoRA SFT for the leak control: train on the eval tasks' own oracles at maximal-memorization settings |
| `scripts/memorization_probe.py` | token-F1 lift on training vs held-out rows — verify a checkpoint actually learned its data before interpreting any benchmark delta |

## The experiment discipline

Screen cheap, confirm before believing, and control the instrument itself:

1. **Memorization probe first** — a flat benchmark delta on a checkpoint that
   never learned its data is a training bug, not a science result.
2. **Screen at 1 attempt/task, confirm at 3** — the ledger documents why
   (single-attempt CIs on a noisy benchmark are ±0.23 wide).
3. **Run the leak control** — train on the eval's own oracles and demand a
   strongly positive delta. If the instrument cannot detect deliberate
   contamination, no negative result from it is interpretable.
