# Contributing to PostTrain Arena

<!-- markdownlint-disable MD013 MD060 -->

Start with one working task, a focused pipeline fix, or a documentation improvement. You can validate task structure without a GPU, provider account, or BenchFlow installation.

- [Contribute environments](#contribute-environments)
- [Legacy skill-package tooling](#legacy-skill-package-tooling)
- [Contribute to the training pipeline](#contribute-to-the-training-pipeline)
- [Prepare a pull request](#prepare-a-pull-request)

The current hosted workflow accepts environment collections and configurable experiments through the headless CLI/API. Model, training data, evaluation data, method, and parameters are configurable now; future fixed hackathon protocols do not gate registration. Structural acceptance does not allocate training or establish a verified result. See [hosted collaboration and execution](docs/hf-submission-lab.md).

## Contribute environments

### 1. Create a team entry

From a fresh clone, run the following at the repository root. Replace `your-team`, `your-env-name`, and the contact details before submitting. If your team already has a manifest, edit it instead of overwriting it.

```bash
mkdir -p submissions/your-team/envs
cp -R starting-kit/template submissions/your-team/envs/your-env-name
cat > submissions/your-team/submission.yaml <<'YAML'
team_name: Your Team
contact_email: you@example.com
track: environments
YAML
```

Use a descriptive package name such as `finance-fed-minutes-classify`. The hosted registry accepts **1–200 environment packages** in a collection. The public repository checker still emits an older draft-minimum warning below 50 and fails above 200; that warning does not impose a 50-package minimum on hosted submissions. See the [collection contract](submissions/README.md).

### 2. Complete the task package

| File or directory | What to implement |
| --- | --- |
| `task.md` | Author metadata, declared resources, and an unambiguous `## prompt` describing the required output |
| `environment/Dockerfile` | Reproducible task image, dependencies, and seed data |
| `verifier/test.sh` and `verifier/test_outputs.py` | Checks of actual trial output; write the reward under `/logs/verifier/` |
| `verifier/verifier.md` and `verifier/rubrics/` | Explain what is scored and why |
| `oracle/solve.sh` | A reference solution that passes the same verifier in the same image |

The template's verifier is a placeholder. Replace it with checks that reject missing, incorrect, and trivial output. Use the [worked examples](starting-kit/README.md) and [full authoring specification](https://posttrain.com/docs/spec) for the task contract. Keep category/modality metadata in frontmatter rather than encoding it into directory names.

### 3. Validate structure and behavior

```bash
# Fast structural checks; Python standard library only.
python3 scripts/check_task.py submissions/your-team/envs
python3 scripts/check_submission.py

# Requires Docker and a running daemon. Oracle must score 1.0.
scripts/run_local.sh submissions/your-team/envs/your-env-name

# Empty trial must not score 1.0; normally it scores 0.0.
scripts/run_local.sh submissions/your-team/envs/your-env-name --skip-oracle
```

Both Docker commands must exit successfully. For `--skip-oracle`, success means the verifier rejected the empty trial. A missing reward file is a harness/verifier failure, not a valid zero score.

| Check | Proves | Does not prove |
| --- | --- | --- |
| `check_task.py` | Required files, selected frontmatter keys, and prompt section exist | Full YAML schema, allowed vocabulary, or instruction quality |
| `check_submission.py` | Manifest, package structure, and upper size bound pass | Hosted registration, runtime correctness, or training allocation |
| Oracle replay | Reference solution receives reward `1.0` | Task difficulty or verifier robustness |
| Empty-trial replay | Doing nothing does not receive full reward | Resistance to other shortcuts, leakage, or reward hacking |

The Docker harness disables trial networking by default. Use `--network` only when the task needs it and document why. Image builds may access the network to install dependencies. On macOS, install GNU coreutils for `gtimeout` if you need the harness's wall-clock cap; without `timeout`/`gtimeout`, the harness warns and continues without that cap.

### 4. Review task quality

Before requesting review, check that:

- Instructions, expected outputs, and scoring agree.
- The oracle works within the declared resources and explains a legitimate solution.
- The verifier checks correctness rather than a fixed filename or exact oracle bytes alone.
- Wrong and incomplete outputs fail; infrastructure failures are distinguishable from model failures.
- Task assets have appropriate licenses, provenance, and no credentials or private data.
- The declared network policy matches actual task needs.

For a public contribution, include both replay results in your pull request. To register the collection in the hosted arena, use its CLI/API to validate and submit the pinned public repository or dataset; a pull request is not required for hosted registration. Follow [the hosted guide](docs/hf-submission-lab.md) for agent onboarding, matching experiment configuration, supported execution profiles, and evidence review. The pinned shift-schedule profile has a [completed, independently replayed example](docs/hf-submission-lab.md#recorded-evidence); that evidence does not make arbitrary collections executable. Deeper difficulty, leakage, and adversarial review remain separate from local structural checks.

## Legacy skill-package tooling

The repository retains a `track: skills` structural format from the earlier competition proposal: packages live at `skills/<skill-name>/SKILL.md`; the checker warns below 20 and rejects more than 100. This is not an active hosted skill-submission or frozen-agent evaluation protocol. The current hosted workflow registers `track: environments`. Preserve existing skill packages when maintaining the repository, but do not infer an executable evaluation path from their structural checks.

## Contribute to the training pipeline

The implementation lives in [`pipelines/benchflow-task-posttrain/`](pipelines/benchflow-task-posttrain/). Read [architecture/status](docs/architecture-status.md) before changing compatibility claims, and the [operator guide](docs/training-pipeline.md) for training behavior.

### Inspect a recipe without training

Python 3.12+ is required. The base package has no runtime dependencies; installing it does not install the GPU training stack.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e pipelines/benchflow-task-posttrain

cd pipelines/benchflow-task-posttrain
posttrainarena-train validate --config configs/qwen3.5-9b-data-agent-canary.toml
posttrainarena-train plan --config configs/qwen3.5-9b-data-agent-canary.toml --run-name contribution-check
posttrainarena-train run --config configs/qwen3.5-9b-data-agent-canary.toml --run-name contribution-check --dry-run
cd ../..
```

These commands validate and plan locally. A dry run writes local reports but does not perform optimizer updates or establish a model score. Actual training needs the dependencies, credentials, serving topology, and compute described in the operator guide.

### Validate pipeline changes

From the repository root in the activated environment:

```bash
python -m pip install -e 'pipelines/benchflow-task-posttrain[test]'
python -m pytest pipelines/benchflow-task-posttrain/tests -q
python -m py_compile pipelines/benchflow-task-posttrain/src/posttrainarena/benchflow_pipeline/*.py
```

The test extra installs substantial dependencies and pinned upstream Git packages; the tests are designed as no-spend contracts. Keep new tests free of paid service calls. New recipes must pin model/dataset revisions, specify task lists, and document expected compute. Preserve module boundaries: BenchFlow owns tasks and sandbox/verifier lifecycle, OpenCode owns agent interaction, and TRL owns optimization. OpenEnv is a separate protocol adapter.

Use the [HF Jobs guide](docs/hf-jobs.md) for the checked-in OpenCode/TRL launcher. The [hosted Agent Collabs workflow](docs/hf-submission-lab.md) is a separate CLI/API implementation with bounded execution profiles; public repository access does not grant Space access or authorize a job. Do not equate completed planner output with training, adapter reload with task success, or seen-task success with held-out generalization.

## Prepare a pull request

1. Fork the repository and create a branch for a focused change.
2. Run the relevant checks above. Documentation-only changes should have working relative links and commands matched to the checked-in implementation.
3. Describe the problem, changed behavior, and exact validation results. State anything you could not run and why.
4. For environment entries, include oracle and empty-trial rewards. For pipeline changes, include tests and recipe validation/dry-run results.
5. Keep generated checkpoints, trajectories, raw provider responses, private artifacts, and secrets out of the diff. Public evidence links should be accessible and describe the precise result they establish.

The `tasks-check` workflow checks package structure and manifests when relevant paths change. The `benchflow-posttrain-pipeline` workflow runs pipeline contract tests, dependency resolution, and CLI smoke checks. Neither is proof of competition-scale training or a sealed-suite result. See the workflow files under [`.github/workflows/`](.github/workflows/).

For website bugs or copy fixes, open an issue here with a public URL and reproduction steps; the website is developed separately. For help, see [SUPPORT.md](SUPPORT.md) or [Discord](https://discord.gg/mZ9Rc8q8W3). Report vulnerabilities using [SECURITY.md](SECURITY.md).

## Licensing and future competition rules

Repository code is [AGPL-3.0](LICENSE) unless otherwise noted. Include source provenance and applicable licenses for contributed assets. Earlier draft competition proposals named CC-BY-4.0 for text/data and Apache-2.0 for code; those proposals do not establish current hosted acceptance, automatic relicensing, or finalized competition terms. Any future fixed event must publish its own release, scoring, and eligibility rules.

<!-- markdownlint-enable MD013 MD060 -->
