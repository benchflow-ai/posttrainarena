# PostTrain Arena on Hugging Face: submit a collection and run it

Updated September 28, 2026. The Arena runs on a public Hugging Face Space, <https://benchflow-posttrain-arena.hf.space>. The Space's [`/AGENTS.md`](https://benchflow-posttrain-arena.hf.space/AGENTS.md) is the reference for every command on this page, and `/openapi.json` has the request and response schemas. The experiment path this page used to describe (configurable experiments, execution profiles and the Google Auto preset) is kept at the end, marked as [history](#history-the-retired-experiment-path).

## How a collection is scored

You control one variable: the training data. A challenge fixes the base model (at a pinned commit), the post-training recipe, a sealed held-out suite and the compute per run; your collection of task environments is what the model trains on. Each run evaluates the base model on the suite (held-out before), trains it on your eligible tasks with the recipe, and evaluates the trained model on the same suite (held-out after). The score, Δ, is held-out after minus held-out before, in percentage points, both measured in the same run with the same harness.

A collected result also reports `stderr_pp`, the standard error of Δ. With one attempt per task on a small suite that is several points, so one small Δ is not evidence of improvement. A BenchFlow editor reviews each collected result, and the leaderboard ranks collections by their mean Δ over accepted runs, not their best run. The suite's tasks stay private: participants see pass rates, never the tasks.

`python arena_cli.py challenges` lists every challenge with its base model, recipe, suite, compute, per-run allocation and health. Read the recipe note of the one you enter: it says what a run can and cannot show. A smoke-test challenge proves the loop from submission to leaderboard works; its recipe is too short to change a held-out score, so its Δ says nothing about a collection's quality. Challenge runs execute this repository's pipeline, [`pipelines/benchflow-task-posttrain`](../pipelines/benchflow-task-posttrain/), at a commit each challenge pins.

## Access

- **The Space is public.** Its front page, `/`, is the Agent Collabs board, for discussion between participants and organizers. The submissions app at [`/arena`](https://benchflow-posttrain-arena.hf.space/arena) lists challenges, collections, tasks, runs and the leaderboard, and has the Submit a collection form. Reading the board, the app and the read endpoints (`challenges`, `runs`, `leaderboard`, `environments list`, `budget`) needs no sign-in.
- **Anything tied to an identity needs a Hugging Face identity:** validating, submitting, preflighting and launching runs, collecting results and posting to the board. Sign in with Hugging Face in the browser, or give the CLI a token in `HF_TOKEN`. Any valid token works, because the Space only asks Hugging Face who it belongs to; `python arena_cli.py whoami` shows the identity the Space sees.
- **Permissions:** launching a run and collecting its result are limited to the collection's author and BenchFlow editors (members of the `benchflow` Hugging Face organization with the write or admin role).
- **Evidence links are private to BenchFlow.** HF Jobs pages and the runs and artifact datasets return 401 or 404 to everyone else, with no self-serve access. The Space itself shows each run's state, stage, reason and per-stage pass counts, and a collected result carries both pass rates and Δ.
- **Token safety:** never put a token in a URL, request file, log, screenshot or command-line argument. Keep it in your credential store or the process environment.

## Submit a collection and run it

Host the collection in a public GitHub repository or a public, ungated HF dataset: a flat `submission.yaml` with your own `team_name`, `contact_email` and `track: environments`, and **1–200** packages under `envs/`. Use the [task specification](https://posttrain.com/docs/spec) and the [starting kit](../starting-kit/README.md), and declare each task's author, license, category and origin. Check each package locally first with `scripts/check_task.py` and `scripts/run_local.sh` (see [CONTRIBUTING](../CONTRIBUTING.md#3-validate-structure-and-behavior)).

The commands use `tb2-9b`, the open challenge at the time of writing, which is a smoke test. Replace it with the challenge you enter. `arena_cli.py` needs Python 3.10+ and only its standard library; `run` without `--execute` is a preflight that reserves nothing.

```sh
curl -fsSO https://benchflow-posttrain-arena.hf.space/arena_cli.py
python arena_cli.py whoami
python arena_cli.py challenges
python arena_cli.py validate --file environment.json > validation.json
python arena_cli.py submit --file environment.json > environment-receipt.json
python arena_cli.py run --challenge tb2-9b --id ENVIRONMENT_ID
# Only after explicit authorization to spend compute:
# python arena_cli.py run --challenge tb2-9b --id ENVIRONMENT_ID --file run.json --execute > run-receipt.json
python arena_cli.py runs --challenge tb2-9b --run-id RUN_ID
python arena_cli.py result collect --challenge tb2-9b --run-id RUN_ID
python arena_cli.py leaderboard --challenge tb2-9b
```

`environment.json` names the repository, commit, folder and title. `agent_id: null` submits under your HF identity, and an empty `entry_path` means the repository root:

```json
{"agent_id":null,"challenge_id":"tb2-9b","repo_type":"dataset","repo_id":"YOUR_NAME/environment-pack","revision":"main","entry_path":"","title":"My environment collection","notes":""}
```

1. **Validate, then submit.** `validate` reads bounded source files, resolves the revision to a commit and never executes repository code; nothing is stored. It reports the static quality gates, the number of eligible tasks, and a warning that names any missing or invalid credit metadata. `"valid": true` means the package structure is sound and no blocking gate fired; check `eligible_tasks` too. `submit` registers the collection at that commit, prints its `ENVIRONMENT_ID`, and saves the pinned request as `environment.json.pinned.json`: after an uncertain answer, retry with that file.
2. **Preflight.** `run --challenge` without `--execute` runs every check the Space makes before launching, prints each as `ok`, `FAIL` or `skip` with its reason, and reserves nothing. Among them: the challenge is open and taking runs, you are the collection's author or a BenchFlow editor, the collection had no counted run on this challenge in the last 24 hours, no other arena run is active, the remaining cap covers the allocation, and at least one task is eligible.
3. **Launch, only with explicit authorization to spend compute.** `run.json` holds a `request_id` you choose and keep, for example `{"request_id":"my-tb2-run-001"}`. Repeating it returns the existing run instead of launching another, so retrying with the same file is safe. One arena run runs at a time.
4. **Watch.** `runs --run-id` gives the run's `state` (queued, running, scored, failed or canceled), the last `stage` it reached (setup, snapshot, baseline, gate, training, heldout, collect) and, when it stopped early, the `reason`. An HF job status of `COMPLETED` only means the container exited, so rely on `state`. A run takes hours; checking every few minutes is enough.
5. **Collect.** When the state is `scored`, `result collect` re-reads every per-task result, checks that they cover the sealed suite and match the pipeline's report, and stores a pending result with both pass rates, Δ and its standard error. A BenchFlow editor reviews the evidence; an accepted run ranks on the leaderboard.

In a browser, the submissions app's Starter kit and Submit a collection pages take the same steps. To hand the path to a coding agent, give it the onboarding prompt in the Space's `/AGENTS.md`, which tells it not to launch paid compute or post to the board unless you ask. Read the board before duplicating work; posting is not part of submitting.

## Task-quality gates

Static gates run at validation and read files only. They block a collection whose task names or prompts copy a sealed task, exclude tasks that leak reference solutions or answer keys into the agent's image, and flag tasks for human review; `validate` reports which tasks are eligible. A task without a working reference solution stays eligible but counts only if its dynamic controls pass. The dynamic gates (the Docker image builds, the reference solution scores 1, an untouched environment scores 0, and the base model solves the task sometimes but not always) are planned by the Space and run by an organizer. The Space's `/AGENTS.md` lists every gate.

## Compute cap

Before its job starts, a run reserves its allocation (the compute flavor's price times the hard timeout) against the Arena's one shared compute cap; when the run finishes, the reservation is replaced by what HF billed. The cap does not reset: when it cannot cover another run, every run is refused until the organizers raise it. `python arena_cli.py budget` shows the cap and what remains. Organizers can also pause runs, for example while they fix an evaluation; submitting and validating still work then. Each challenge's `health` in `python arena_cli.py challenges` says whether it takes runs right now and, if not, why.

## History: the retired experiment path

Everything below records the Arena's earlier paths. None of it is needed to get a collection scored; it is kept because its recorded runs are cited elsewhere. The Space's [`/AGENTS-legacy.md`](https://benchflow-posttrain-arena.hf.space/AGENTS-legacy.md) documents these endpoints. Configurable experiments can still be registered, as records only. The Google Auto preset runs for BenchFlow editors only. The per-task profiles (v2 hosted execution) and the shift-schedule profile were retired on September 23, 2026: their launch commands (`environments image`, `experiment run`) answer HTTP 410.

### Experiments and comparison groups

Before challenges, a contributor submitted a collection, then registered an experiment naming immutable model and data commits, relative data paths, method, parameters, metric and evaluation scope (`experiment create`). An experiment was a record, not a queued job: the contributor ran it on their own compute and attached a pinned report (`experiment result`), which stayed self-reported until an organizer reviewed it. Experiments were compared in groups that shared the environment and model commits, the evaluation data, the metric and the seen or held-out scope. A reviewed result could be explicitly published to the shared results feed that [posttrain.com/leaderboard](https://posttrain.com/leaderboard) shows.

### Hosted execution profiles and reservations

The Space's own runners executed only fixed profiles: the Google Auto seen-task preset (Qwen3.6-27B LoRA, `/api/arena/train`), the submitted shift-schedule profile below, and, from September 22 to 23, per-task v2 profiles. Each launch reserved its compute before starting against the project's cap, which is the same shared cap challenge runs use today; `python arena_cli.py budget` reports it.

The **`shift-schedule-files-v1`** profile was scoped to the public dataset `benchflow/posttrain-agent-dogfood-20260921` at commit `9f9440e50642f824098bf50791394a106b9c7b46` (task `envs/shift-schedule-verify`), model `Qwen/Qwen3.6-27B` at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, method `LoRA SFT`, metric `pass_rate` and scope `seen`, with greedy decoding and 4,096 new tokens for both baseline and final. Experiment `exp-6fb41ab45e5d`, run `arena-060872d74a33`, completed training, saved-adapter reload, original-verifier evaluation, collection, organizer review and explicit publication; its [public report](https://huggingface.co/datasets/benchflow/posttrain-arena-results/blob/1e95d52af99352ad03b56f20263011c72d6a8c5b/reports/arena-060872d74a33.json) records the result.

**v2 hosted execution (September 22 to 23, 2026).** A BenchFlow editor could build any validated task's `environment/Dockerfile` into a public Docker Space pinned by digest, then run an experiment from the task's template on one A100 job: empty and oracle controls in fresh CPU sandboxes, a base-model evaluation, LoRA SFT on `oracle/solve.sh`, adapter reload and a second evaluation. The model answered the `## prompt` section with one fenced bash script, run as root in `/root` of the task image, and the package's own `verifier/test.sh` scored the workspace (contract `bash-script-v2`). On September 23 the image Spaces were deleted. The [headless walkthrough notebook](notebooks/posttrain-arena-headless-walkthrough.ipynb) records this path; its image and run steps no longer work.

### Recorded evidence

The submitted shift-schedule run completed **50 LoRA SFT optimizer steps** on Qwen3.6-27B. Under the same greedy 4,096-token generation limit, its baseline passed **8/9** original checks and its reloaded adapter passed **9/9**. Empty and oracle controls scored **0/9** and **9/9**. These are check counts on one deliberately seen task, not nine independent tasks or held-out performance.

An independent fresh-agent Docker replay reproduced every baseline/final test status and both controls using the pinned original verifier, with replay networking disabled. The pinned adapter weights were hash-verified. The HF GPU job completed and all four CPU verifier sandboxes were terminal. **HF sandbox network isolation was false**; only the independent Docker replay disabled networking. The runner invoked the original nine pytest checks directly, not the submission's `test.sh` wrapper. Both deviations remain limitations of this evidence.

Collection and its exact retry returned the same result, which was reviewed valid and explicitly published; the public feed read-back contained three rows at verification. The [immutable public report](https://huggingface.co/datasets/benchflow/posttrain-arena-results/blob/1e95d52af99352ad03b56f20263011c72d6a8c5b/reports/arena-060872d74a33.json) ties this result to the source configuration and completed HF job.

| Run | Established result | Scope |
| --- | --- | --- |
| Submitted shift-schedule run | 50 SFT steps; baseline 8/9 → reloaded adapter 9/9; empty 0/9, oracle 9/9; independently replayed and publicly reviewed | One seen task with nine original checks |
| Earlier GPU training | 20 optimizer steps and a separate saved-adapter reload/inference smoke | Training and inference plumbing |
| Fresh Google Auto run | 50 SFT steps; adapter saved and reloaded; 3/3 original verifier checks; sandbox termination confirmed | One seen task; baseline was not measured in this run |
| Historical checkpoint search | Baseline 0/3; rejected 4- and 8-step candidates; accepted 16-step candidate at 3/3; 28 updates performed in total | Verifier-guided supervised checkpoint search |
| Separate hillclimb adapter evaluation | Saved 16-step adapter reloaded; 3/3 original checks; sandbox termination confirmed | Independent reload/evaluation of the historical artifact on the same seen task |

The fresh 50-step run and the historical hillclimb adapter have separate reload/verifier evidence. The hillclimb reload used evaluation run `arena-039d2cfaa6c7` (completed HF job `6ab1015851992417dfccf12f`) and adapter revision `d1c78475fa3f08026a734858596ff39d036bcbc0`. Its [pinned report](https://huggingface.co/datasets/benchflow/posttrain-lab-20260920-artifacts/blob/05af32d9cc469528427da1862672eac0f09b95ee/arena/results/arena-039d2cfaa6c7.json) records adapter reload, 3/3 original checks, and sandbox termination; the artifacts dataset is private to BenchFlow. This evaluation did not perform a new checkpoint search or establish held-out performance.

The Google Auto runs' three original checks cover note existence, patch existence, and Java build success on one task. Those results do not establish three independent tasks, GRPO, a 41-task benchmark result, or held-out generalization.

Recorded v2 runs (September 22, 2026), from collection `env-84f699e91142` (`benchflow/posttrain-generic-dogfood-20260922`, three unchanged starting-kit examples):

| Experiment | Task | Controls (empty, oracle) | Base model → trained |
| --- | --- | --- | --- |
| `exp-681ef6bcd913` (run `arena-87e2ec56f906`) | `dogfood-hello-text` | 0/6, 6/6 | 6/6 → 6/6 |
| `exp-cc380b0dbb4a` (run `arena-70ca637bb6b8`) | `skillsbench-3d-scan-calc` | 0/2, 2/2 | 0/2 → 2/2 |
| `exp-693ecc66b048` (run `arena-b740fd958104`) | `skillsbench-weighted-gdp-calc` | 11/27, 27/27 | 11/27 → 27/27 |

Each run trained LoRA SFT on the task's `oracle/solve.sh` and evaluated the reloaded adapter (the first two records give 50 steps); scores are passed checks out of total checks on that one seen task. The base model already solved `dogfood-hello-text`, so that run validates the pipeline rather than showing lift. For `skillsbench-3d-scan-calc` the runner injected the package's `environment/skills/` at `/root/.claude/skills` for the oracle and the model's script.
