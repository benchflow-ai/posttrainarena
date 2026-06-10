---
version: "1.0"
metadata:
  author_name: Your Name
  author_email: you@example.com
  category: natural-science   # one of the eight domains — see /docs/spec
  difficulty: medium          # easy | medium | hard
  tags: [your, tags, here]
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 180
environment:
  build_timeout_sec: 600
  cpus: 1
  memory_mb: 2048
  storage_mb: 10240
  allow_internet: false
---

## prompt

State the task here in one short paragraph. The agent reads this first.

Then any structured detail it needs — file paths, expected output
schema, evaluation rules. Be specific: the verifier is mechanical, so
ambiguous prompts produce ambiguous trials.

Required output: write your final answer to `/root/answer.json` (or
whichever path your verifier reads). The verifier in this template is a
placeholder; replace it with real checks before submitting.
