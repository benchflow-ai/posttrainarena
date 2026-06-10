# Plain Ubuntu base — the dogfooded SkillsBench tasks use the same starting
# point. Add only what is task-specific.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Install task-specific Python deps. Pin everything.
# RUN pip3 install --break-system-packages \
#     requests==2.32.3

# Copy any seed data the agent reads at trial time.
# COPY data.json /root/data.json
