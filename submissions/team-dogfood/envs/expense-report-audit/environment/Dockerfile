FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

# Verifier deps preinstalled at build time so verification works with
# allow_internet: false (the verifier shim falls back to a network
# install of uv only when pytest is absent).
RUN pip install --no-cache-dir \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5

RUN mkdir -p /logs/verifier /logs/agent

# Seed data only. The expense CSV the agent must audit is placed at
# /root/expenses.csv. The verifier carries its own pristine copy
# (verifier/data/expenses.csv) and recomputes the expected audit from it,
# so the ground truth is never derivable from anything in this image.
COPY seed/expenses.csv /root/expenses.csv
