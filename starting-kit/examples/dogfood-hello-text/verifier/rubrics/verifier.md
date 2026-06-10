# dogfood-hello-text rubric

Reward 1.0 when `/root/answer.json` exists, parses as JSON, and
contains exactly `greeting="hello"`, `subject="post-training"`, and
`year=2026`. Reward 0.0 otherwise. No partial credit — every pytest
test must pass.
