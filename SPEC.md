# task.md specification

How to author a PostTrain Arena submission. This is the reference; the body of
each section is what reviewers compare against. The same content is rendered at
[posttrain.com/docs/spec](https://posttrain.com/docs/spec).

A submission is four pieces in one directory: a `task.md`, an `environment/`, a
`verifier/`, and an `oracle/`.

## Contents

- [Overview](#overview)
- [Submission layout](#submission-layout)
- [task.md](#taskmd)
- [environment/](#environment)
- [verifier/](#verifier)
- [oracle/](#oracle)
- [Validate locally](#validate-locally)
- [Submit](#submit)

## Overview

PostTrain Arena tasks are *self-contained*: a fresh checkout plus Docker is
enough to build the environment, run an agent, and score the result. There is no
hidden runtime config; every limit and scoring rule lives in the submission
directory.

The format is built on `task.md` — a single human-authored file with YAML
frontmatter (limits, metadata, optional multi-agent structure) and a Markdown
body (the prompt, plus optional per-scene or per-role guidance). It expands
through defaults into the same canonical contract that the eval pipeline
consumes, so a minimal author file stays small but a richer one can spell out
scenes, roles, and a simulated user without leaving the file.

## Submission layout

```
tasks/
└─ your-task-name/
   ├─ task.md                    # required — frontmatter + prompt
   ├─ environment/
   │  ├─ Dockerfile              # required — built fresh per task
   │  ├─ <seed-data>             # optional — any files the agent reads
   │  └─ skills/                 # optional — agent-discoverable skill packages
   ├─ verifier/
   │  ├─ test.sh                 # required — pytest runner shim (boilerplate)
   │  ├─ test_outputs.py         # required — the actual checks
   │  ├─ verifier.md             # required — rubric declaration (frontmatter)
   │  └─ rubrics/
   │     └─ verifier.md          # required — plain-language pass criteria
   └─ oracle/
      └─ solve.sh                # required — produces a passing trial
```

Naming convention: `<env-or-domain>-<short-description>` — for example
`gmail-workflow-delegation` or `finance-weighted-gdp-calc`. Category, modality,
and any safety qualifier live in the frontmatter, not the directory name.

The fastest way to start is `cp -R tasks/template tasks/your-task-name` from a
fresh checkout. The three `skillsbench-*` tasks in the repo are the working
references — copy the parts you need from them.

## task.md

Two halves separated by a `---` fence: YAML frontmatter on top, Markdown body
below. Frontmatter declares limits and metadata; the body holds the prompt and,
optionally, per-scene or per-role instructions and a user persona.

### Minimal example

```yaml
---
version: "1.0"
metadata:
  author_name: Ada Lovelace
  author_email: ada@example.com
  category: sciences
  difficulty: medium
  tags: [calculation, spreadsheet]
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 180
environment:
  cpus: 1
  memory_mb: 4096
---

## prompt

State the task here. One short paragraph that the agent will read first,
followed by any structured detail it needs.
```

### Frontmatter reference

All fields are optional unless marked required. Limits default to safe values
(verifier 180s, agent 900s, 1 CPU, 4 GB RAM). Field names use snake_case
throughout.

- `version` (required) — schema version, currently `"1.0"`.
- `metadata.author_name` + `author_email` (required) — contact for the contributor.
- `metadata.category` (required) — one of the eight domains, e.g. `sciences`, `cybersecurity`, `office-knowledge-work`.
- `metadata.difficulty` — `easy` | `medium` | `hard`.
- `metadata.tags` — short string list, used for routing and filtering.
- `agent.timeout_sec` — wall-clock budget for the agent.
- `verifier.timeout_sec` — budget for the scoring run after the agent finishes.
- `environment.cpus` / `memory_mb` / `storage_mb` — sandbox limits.
- `environment.network_mode` — `no-network` (default), `allowlist` with `allowed_hosts`, or `public`.
- `agents.roles` — optional named roles when the task is multi-agent (see below).
- `scenes` — optional ordered list of scenes, each referencing one or more roles.
- `user` — optional simulated-user model + stop rule when the agent needs a counterpart.

### Body sections

The body is Markdown with a small set of well-known headings. Anything not
matching a known heading is passed through as authoring notes and ignored by the
runtime.

- `## prompt` (required) — what the agent sees first.
- `## scene:<name>` — guidance shown when that scene starts, if you declared `scenes`.
- `## role:<name>` — guidance shown to that role, if you declared `agents.roles`.
- `## user-persona` — the simulated user's mindset, if you declared `user`.

### Multi-agent example

The richer form composes roles, scenes, and a simulated user in one file — the
same authoring document the eval pipeline consumes directly:

```yaml
---
version: "1.0"
metadata:
  author_name: benchflow
  category: office-knowledge-work
  difficulty: hard
  tags: [multi-agent, planning]
agent:
  timeout_sec: 1200
verifier:
  timeout_sec: 240
environment:
  cpus: 2
  memory_mb: 4096
agents:
  roles:
    planner:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
    executor:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: high
scenes:
  - name: plan
    turns: [{ role: planner }]
  - name: implement
    turns: [{ role: executor }]
user:
  model: claude-haiku
  stop_rule: satisfied-or-5-rounds
---

## prompt

Refactor the tiny service so it keeps the same public behavior while splitting
request parsing, business logic, and output formatting into separate modules.

## scene:plan

Read the task, inspect the code, and write a concise implementation plan.

## scene:implement

Apply the plan. Run the verifier before finishing.

## user-persona

You are impatient and only reveal the order id if the agent asks for it
specifically.
```

## environment/

The environment directory holds the Dockerfile and any seed data the agent needs
at trial time. Start from a plain Ubuntu base and add only what the task needs —
the dogfooded SkillsBench tasks all begin with `FROM ubuntu:24.04`.

```dockerfile
# environment/Dockerfile
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python + anything else the task needs.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Task-specific deps. Pin versions.
RUN pip3 install --break-system-packages \
    requests==2.32.3

# Seed data the agent reads at trial time.
COPY data.json /root/data.json

# Optional: bundle a skill package the agent can discover.
# COPY skills /root/skills
```

The agent works inside `$BENCHFLOW_WORKSPACE` (default `/root`). Anything the
verifier needs to see must be written under that path before the agent exits.
Trial sandboxes start fresh per attempt.

An optional `environment/skills/` directory follows the
[Agent Skills spec](https://agentskills.io/home): each subdirectory is one skill
with its own `SKILL.md`, scripts, and references. Bundling skills with the task
is the usual way to teach an agent the domain-specific moves your verifier
expects.

## verifier/

The verifier runs after the agent finishes and writes three artifacts to
`/logs/verifier/`:

- `reward.txt` — a single float, one of `0.0` or `1.0` for binary checks (or a real number in [0,1] for graded ones).
- `reward.json` — `{ "reward": <float> }`.
- `ctrf.json` — pytest's [CTRF](https://ctrf.io/) details for per-check breakdowns.

`verifier/test.sh` is a small shim that runs pytest against `test_outputs.py` and
writes those artifacts — copy it verbatim from the template. Put your real checks
in `test_outputs.py`:

```python
# verifier/test_outputs.py — pytest tests the trial's outputs.
# Passing → reward 1.0; any failure → reward 0.0. Use one test class
# per check group so partial-credit cases stay readable.
import json, os
from pathlib import Path
import pytest

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
ANSWER_FILE = WORKSPACE / "answer.json"

class TestAnswerFileExists:
    def test_file_exists(self):
        assert ANSWER_FILE.exists(), f"Answer file not found at {ANSWER_FILE}"

    def test_file_is_valid_json(self):
        with open(ANSWER_FILE) as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Answer file is not valid JSON: {e}")

class TestAnswerContent:
    def test_result_matches_expected(self):
        with open(ANSWER_FILE) as f:
            data = json.load(f)
        assert data["result"] == 42, f"Expected 42, got {data['result']}"
```

The `verifier.md` file alongside the tests declares the rubric the pytest results
roll up into, and gives reviewers a plain-language description for edge cases.
Schema:

```yaml
---
document_version: '0.3'
verifier:
  name: your-task-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      correctness:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

State what a passing trial looks like in plain language. Human reviewers
read this when grading edge cases the pytest tests can't decide alone.
```

The verifier runs in its own container by default with no network access.
Declare `verifier.network_mode: public` in the frontmatter only if the verifier
calls an LLM judge or external API.

## oracle/

The oracle is a reference solution. Its job is to prove the task is reachable:
when run under the same environment, it must score a passing reward (1.0 for
binary checks). Reviewers re-run the oracle as part of acceptance, and CI re-runs
it on every change to the environment image.

```bash
#!/bin/bash
# oracle/solve.sh — reference solution. Runs in the same container the
# agent uses. Must achieve a passing score so reviewers can confirm the
# task is solvable; CI re-runs it on every image bump.
set -e
WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
mkdir -p "$WORKSPACE"

cat > "$WORKSPACE/answer.json" << 'JSON'
{
  "result": 42
}
JSON
```

Keep the oracle as simple as the task allows — it is documentation as much as a
regression check. A long, clever oracle usually means the task is doing too much.

## Validate locally

Use the `bench` CLI to validate before opening a PR. Two levels matter for
authors:

```bash
# 1. Schema-only — fast, no Docker required
bench tasks check ./tasks/your-task-name --level schema

# 2. Publication-grade — builds the image, runs the oracle, scores it
bench tasks check ./tasks/your-task-name --level publication-grade
```

Schema mode catches frontmatter typos, missing required fields, and unknown
headings. Publication-grade additionally verifies the full submission contract —
including that `verifier/rubrics/` contains at least one rubric file. CI runs
schema on every PR and publication-grade on PRs that touch `tasks/`; running the
same command locally first cuts the review loop.

Install the CLI once:

```bash
# Recommended: uv tool install (isolated, fast)
uv tool install benchflow

# Or with pip
pip install benchflow
```

## Submit

Open a pull request against this repository with your task directory under
`tasks/`. Include the local `publication-grade` output in the PR description so
reviewers know it built and passed against your machine. Discussion happens on
[Discord](https://discord.gg/mZ9Rc8q8W3) and in the PR thread.

Accepted tasks are released openly. We post-train a Qwen3-8B model across the
accepted pool and score it on the private IndexBench suite. Standings will be
published once Phase 1 begins.
