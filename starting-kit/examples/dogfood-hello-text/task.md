---
version: "1.0"
metadata:
  author_name: Xiangyi Li
  author_email: xiangyi@benchflow.ai
  category: software-engineering
  difficulty: easy
  # This demo's required output is given verbatim in the prompt by design
  # (a "write exactly this object" task) — there is no held-out answer, so the
  # managed pipeline's answer-leakage audit does not apply.
  output_fully_specified: true
  tags: [dogfood, hello-world, file-io]
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 60
sandbox:
  build_timeout_sec: 300
  cpus: 1
  memory_mb: 1024
  storage_mb: 2048
  allow_internet: false
---

## prompt

Write a JSON file at `/root/answer.json` containing the following exact
object:

```json
{
  "greeting": "hello",
  "subject": "post-training",
  "year": 2026
}
```

The verifier loads the file, parses it as JSON, and checks the three
fields. The file must be valid JSON; whitespace and key order don't
matter.

This is the dogfood task — a fresh contributor uses it to prove the
template + spec + validation pipeline actually work end-to-end. Keep it
this simple: any real submission should be doing something
substantively harder.
