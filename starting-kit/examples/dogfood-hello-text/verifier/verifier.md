---
document_version: '0.3'
verifier:
  name: dogfood-hello-text-verifier
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

A passing trial creates `/root/answer.json` whose parsed contents equal
`{"greeting": "hello", "subject": "post-training", "year": 2026}`. Key
order, surrounding whitespace, and trailing newline are irrelevant;
extra keys are ignored; missing or mismatched values fail the trial.
