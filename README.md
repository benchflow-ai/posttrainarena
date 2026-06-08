# PostTrain Arena

**The open arena for post-training.**

Contribute an RL environment. We post-train a model on it and score what
generalizes across eight under-served domains.

[Website](https://posttrain.com) · [Full spec](https://posttrain.com/docs/spec) · [Browse the catalog](https://posttrain.com/catalog) · [Discord](https://discord.gg/mZ9Rc8q8W3)

---

## What this is

PostTrain Arena is an open competition. Anyone can contribute a self-contained
reinforcement-learning environment: a task an agent attempts, a sandbox it runs
in, and a verifier that scores the result. Accepted tasks are released openly.
We then post-train a model across the accepted pool and measure what generalizes
to a private held-out suite — so contributions are rewarded for teaching
transferable skills, not for overfitting a single benchmark.

## How it works

1. **Contribute.** Open a pull request with a task package (see below).
2. **Post-train.** We post-train a Qwen3-8B model across the accepted task pool.
3. **Score.** We evaluate the trained model on IndexBench, a private held-out
   suite where no single domain exceeds 20% of the tasks.
4. **Rank.** Standings are published once Phase 1 begins.

## The eight under-served domains

Every task declares one `category`. We are deliberately seeking coverage in
domains the open ecosystem under-serves:

| Domain | `category` |
| --- | --- |
| Sciences | `sciences` |
| Industrial & Energy Operations | `industrial-energy` |
| Cybersecurity | `cybersecurity` |
| Finance & Economics | `finance` |
| Office & Knowledge Work | `office-knowledge-work` |
| Media & Multimodal Content | `media` |
| AI/ML & Agentic Systems | `aiml` |
| Software Engineering | `swe` |

## Contribute a task

A submission is four pieces in one directory:

```
tasks/
└─ your-task-name/
   ├─ task.md          # frontmatter (limits, metadata) + the prompt
   ├─ environment/     # Dockerfile + any seed data the agent reads
   ├─ verifier/        # scoring logic + a plain-language rubric
   └─ oracle/          # a reference solution that proves the task is solvable
```

- **`task.md`** — a single human-authored file: YAML frontmatter for limits and
  metadata, a Markdown body for the prompt (and optional multi-agent scenes,
  roles, and a simulated user).
- **`environment/`** — a sealed `Dockerfile` built fresh per task, plus any seed
  files. Start from a plain base and add only what the task needs.
- **`verifier/`** — runs after the agent finishes and emits a reward in `[0,1]`,
  plus a rubric reviewers read for edge cases.
- **`oracle/`** — a reference solution that achieves a passing reward, so
  reviewers (and CI) can confirm the task is solvable.

The fastest way to start is to scaffold one:

```bash
bench tasks init tasks/your-task-name
```

The full authoring reference — frontmatter schema, body sections, verifier and
oracle contracts — is in **[SPEC.md](SPEC.md)** (and rendered at
[posttrain.com/docs/spec](https://posttrain.com/docs/spec)).

## Validate locally

Use the `bench` CLI to validate before opening a PR:

```bash
# Schema-only — fast, no Docker required
bench tasks check ./tasks/your-task-name --level schema

# Publication-grade — builds the image, runs the oracle, scores it
bench tasks check ./tasks/your-task-name --level publication-grade
```

Install it once:

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

Accepted tasks are released openly.

## Links

- Website — https://posttrain.com
- Task spec — https://posttrain.com/docs/spec
- Environment catalog — https://posttrain.com/catalog
- Discord — https://discord.gg/mZ9Rc8q8W3

## License

See [LICENSE](LICENSE).
