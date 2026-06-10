FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

# Verifier deps preinstalled at build time so verification works with
# allow_internet: false (the verifier shim falls back to a network
# install of uv only when pytest is absent).
RUN pip install --no-cache-dir \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5

RUN mkdir -p /logs/verifier /logs/agent /var/log

# Seed data only. The deterministic generator (gen_auth_log.py) is
# intentionally NOT copied in — the expected findings must not be
# derivable from anything in the image except the log itself.
COPY seed/auth.log /var/log/auth.log
