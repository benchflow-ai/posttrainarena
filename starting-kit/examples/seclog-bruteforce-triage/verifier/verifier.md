---
document_version: '0.3'
verifier:
  name: seclog-bruteforce-triage-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      bruteforce_triage_correctness:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

A passing trial writes `/root/findings.json` whose content equals the
mechanical application of the stated rules to `/var/log/auth.log`: every
source IP with >= 10 `Failed password for` lines is listed with exact
failure counts, invalid-user counts, sorted target usernames, first/last
failure line numbers, and a compromise verdict that only counts an
`Accepted password for` line occurring after that IP's 10th failure. The
verifier recomputes all expected values from its own pristine copy of
the seed log (`verifier/data/auth.log`) — no expected answer is stored
anywhere the agent can read.
