# Small Python base; the task needs only the standard library (json).
FROM python:3.11-slim

WORKDIR /root

# Verifier deps preinstalled at build time so verification works with
# allow_internet: false (the verifier shim falls back to a network
# install of uv only when pytest is absent).
RUN pip install --no-cache-dir \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5

# Seed data, generated deterministically by generate_data.py (seeded RNG;
# re-running reproduces identical bytes). The image ships only the inputs:
# the scheduling instance and the broken schedule under review. No expected
# answer exists anywhere in the image — the verifier recomputes everything
# from its own pristine copies of these two files.
COPY instance.json /root/instance.json
COPY broken_schedule.json /root/broken_schedule.json
