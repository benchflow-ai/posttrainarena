# Small Python base; the task needs only the standard library (json).
FROM python:3.11-slim

WORKDIR /root

# Seed data, generated deterministically by generate_data.py (seeded RNG;
# re-running reproduces identical bytes). The image ships only the inputs:
# the scheduling instance and the broken schedule under review. No expected
# answer exists anywhere in the image — the verifier recomputes everything
# from its own pristine copies of these two files.
COPY instance.json /root/instance.json
COPY broken_schedule.json /root/broken_schedule.json
