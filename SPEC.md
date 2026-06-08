# task.md specification

How to author a PostTrain Arena submission. A task packages an instruction, a
sandboxed environment, and a verifier into a directory that we run and score
automatically. Authoring writes one `task.md`.

This is the reference; the body of each section is what reviewers compare
against.

## Contents

- [Directory layout](#directory-layout)
- [task.md](#taskmd)
- [environment/](#environment)
- [verifier/](#verifier)
- [oracle/](#oracle)
- [CLI](#cli)
- [Submit](#submit)

## Directory layout

```
my-task/
├── task.md                # config + prompt + optional roles/scenes/user
├── environment/
│   └── Dockerfile         # sandbox image
├── verifier/
│   ├── verifier.md        # optional — selects the verifier strategy
│   └── test.sh            # the script verifier (writes the reward)
└── oracle/                # optional — reference solution proving solvability
    └── solve.sh
```

`verifier/` may also include `test_outputs.py` (a pytest module called by
`test.sh`), rubric files, and judge prompts.

Scaffold a new task with `bench tasks init my-task` (see [CLI](#cli)). It writes
`task.md`, `environment/Dockerfile`, `verifier/test.sh`, `verifier/verifier.md`,
`verifier/rubrics/`, and `oracle/solve.sh`, with `[REPLACE: ...]` markers so
`bench tasks check` stays red until the package has real semantics.

## task.md

A single file: YAML frontmatter (config) on top, a Markdown body (the prompt and
optional role / scene / user-persona guidance) below.

```md
---
schema_version: "1.3"
metadata:
  author_name: alice
  category: sciences
  difficulty: easy
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  network_mode: no-network
  cpus: 1
  memory_mb: 2048
  storage_mb: 10240
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
---

## prompt

Create the requested files in `/app`.

## role:solver

You are responsible for the implementation.

## scene:solve

Solve the task end to end.
```

### Frontmatter

Unknown config keys fail validation; arbitrary labels belong under `metadata`.

- `schema_version` (required) — currently `"1.3"`.
- `metadata.author_name` — contact for the contributor.
- `metadata.category` — one of the eight PostTrain Arena domains, e.g. `sciences`, `cybersecurity`, `office-knowledge-work`.
- `metadata.difficulty` — `easy` | `medium` | `hard`.
- `metadata.tags` — short string list, used for routing and filtering.
- `agent.timeout_sec` — wall-clock budget for the agent.
- `verifier.timeout_sec` — budget for the scoring run after the agent finishes.
- `environment.network_mode` — `no-network` (default), `allowlist` (with `allowed_hosts`), or `public`.
- `environment.cpus` / `memory_mb` / `storage_mb` — sandbox limits.
- `environment.env` — host variables to inject, e.g. `OPENAI_API_KEY: ${OPENAI_API_KEY}`.
- `agents.roles` — named roles when the task is multi-agent.
- `scenes` — ordered list of scenes, each referencing one or more `roles`.
- `user` — simulated-user model + stop rule when the agent needs a counterpart.

### Body sections

The body is Markdown with a small set of well-known headings. Anything not
matching a known heading is passed through as authoring notes.

- `## prompt` (required) — what the agent sees first. State the goal in the first sentence; name exact files/paths; specify constraints. Don't mention the verifier or `reward.txt`.
- `## role:<name>` — guidance shown to that role, if you declared `agents.roles`.
- `## scene:<name>` — guidance shown when that scene starts, if you declared `scenes`.
- `## user-persona` — the simulated user's mindset, if you declared `user`.

### Minimal authoring presets

You can start from a tiny document and expand it into the canonical contract
with `bench tasks normalize`:

```md
---
preset: [code-change]
name: my-namespace/my-task
image: ubuntu:24.04
verifier: verifier/
oracle: oracle/
---

Implement the requested change in `/app`.
```

Presets are authoring sugar that `bench tasks normalize` expands and then
discards. Available presets include `code-change`, `reward-kit`,
`acceptance-live`, `multi-agent`, and `leaderboard-local`. Run
`bench tasks normalize my-task/ --write` to expand in place, or without
`--write` to print the expanded `task.md`.

## environment/

The environment directory holds the `Dockerfile` and any seed data the agent
reads at trial time. The agent works in `/app`. Start from a plain base and add
only what the task needs:

```dockerfile
# environment/Dockerfile
FROM ubuntu:24.04
RUN apt-get update -qq \
 && apt-get install -y -qq python3 curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
```

Anything the verifier needs to see must be written under `/app` before the agent
exits. Trial sandboxes start fresh per attempt.

### Multi-container tasks

A task may ship `environment/docker-compose.yaml` alongside the `Dockerfile`.
The agent always runs in the `main` service; additional services become sibling
containers on the same network — useful for CVE-style tasks where the agent
attacks a separate target.

```yaml
# environment/docker-compose.yaml
services:
  main: {}            # agent container — limits/image injected
  target:
    image: vulhub/struts2-s2-001:latest
    expose: ["8080"]
```

`main` reaches `target` by service name (`http://target:8080`). A verifier can
inspect target-side state by setting `[verifier].service: target`, which runs
`test.sh` inside that container. `environment/Dockerfile` is always required,
even when `main` uses a prebuilt `image:`.

## verifier/

The runtime copies `verifier/` to `/verifier/` and runs it after the agent
finishes. With no `verifier.md`, it runs `/verifier/test.sh`. With a
`verifier.md`, it runs the selected strategy: `script` (a declared command),
`llm-judge`, `agent-judge`, or `reward-kit`.

**Your verifier must write a reward artifact.** Script verifiers write a single
float in `[0.0, 1.0]` to `/logs/verifier/reward.txt`; strategies may also write
`reward.json` and `reward-details.json`. A nonzero exit with no fresh reward
file is treated as infrastructure failure.

| Path | Contents |
|---|---|
| `/app/` | Agent's working directory |
| `/verifier/` | Your `verifier/` directory |
| `/oracle/` | `oracle/` files (oracle runs only) |
| `/logs/verifier/` | Write `reward.txt` (and optionally `ctrf.json`) here |

### Pure bash

```bash
#!/bin/bash
REWARD=0
if [ -f /app/hello.txt ] && [ "$(tr -d '\n' < /app/hello.txt)" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
```

### pytest

```bash
#!/bin/bash
curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
source $HOME/.local/bin/env

uvx \
  --with pytest==8.4.1 \
  --with pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA

if [ $? -eq 0 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
```

### Partial credit

```bash
python3 -c "print($PASSED / $TOTAL)" > /logs/verifier/reward.txt
```

**Security:** don't let the agent write to `/logs/verifier/reward.txt` or modify
`/verifier/test.sh`. For tasks running arbitrary code, set
`network_mode: no-network` and verify output files only.

## oracle/

A reference solution that proves the task is solvable. With
`bench eval create --agent oracle`, the runtime copies `oracle/` to `/oracle/`
and runs `oracle/solve.sh` instead of an agent. It must score a passing reward.
`solve.sh` has the same filesystem access as the agent — write only to `/app/`.
Reviewers and CI re-run it on every environment change.

```bash
#!/bin/bash
echo "Hello, world!" > /app/hello.txt
```

Keep the oracle as simple as the task allows — it is documentation as much as a
regression check.

## CLI

Use the `bench` CLI to scaffold, validate, and run tasks:

```bash
# Scaffold a new task
bench tasks init my-task
bench tasks init my-task --no-pytest --no-oracle

# Expand a minimal task.md into the canonical contract
bench tasks normalize tasks/my-task/ --write

# Validate — schema, structure, runtime support, publication shape
bench tasks check tasks/my-task/ --level schema
bench tasks check tasks/my-task/
bench tasks check tasks/my-task/ --level publication-grade --sandbox docker
bench tasks check tasks/my-task/ --level acceptance

# Confirm the oracle scores reward = 1.0
bench eval create --tasks-dir tasks/my-task/ --agent oracle --sandbox docker

# Run a real agent
bench eval create --tasks-dir tasks/my-task/ --agent gemini --sandbox daytona
```

`--level schema` checks only the authoring entrypoint and prompt parse.
`--level publication-grade` checks the native package contract for registry-ready
authoring (native `oracle/`, `verifier/verifier.md`, rubric files, selected
strategy artifacts, and an explicit reward-JSON output contract).
`--level acceptance` adds static evidence checks for oracle proof, verifier
stability, and calibration; `--level acceptance-live --sandbox <backend>` runs
that evidence through a real sandbox. CI runs schema on every PR and the higher
levels on PRs that touch `tasks/`.

Install the CLI once:

```bash
uv tool install benchflow   # recommended (isolated, fast)
# or
pip install benchflow
```

## Submit

Open a pull request against this repository with your task directory under
`tasks/`. Include the local `publication-grade` output in the PR description so
reviewers know it built and passed on your machine. Discussion happens on
[Discord](https://discord.gg/mZ9Rc8q8W3) and in the PR thread.

Accepted tasks are released openly. We post-train a Qwen3-8B model across the
accepted pool and score it on the private IndexBench suite. Standings will be
published once Phase 1 begins.
