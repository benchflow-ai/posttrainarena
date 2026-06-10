# Small Python base; everything the verifier needs is installed at build
# time so no network access is required during the trial or verification.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Verifier toolchain, baked in at build time. Pin everything.
RUN pip install --no-cache-dir \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5

WORKDIR /root

RUN mkdir -p /root/subtitles /logs/verifier /logs/agent /logs/artifacts

# Seed data: generated deterministically at authoring time by
# environment/generate_seed.py (fixed RNG seed; the generator itself is
# deliberately NOT copied into the image, and no answer key exists in
# the image — the verifier recomputes expectations from its own pristine
# copy of this file).
COPY media/input.srt /root/subtitles/input.srt
