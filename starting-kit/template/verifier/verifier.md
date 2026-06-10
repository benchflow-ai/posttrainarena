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

State what a passing trial looks like in plain language. This is the
spec a human reviewer compares against when grading edge cases the
pytest tests can't decide on their own.
