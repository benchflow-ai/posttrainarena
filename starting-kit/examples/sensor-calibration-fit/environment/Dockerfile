FROM python:3.11-slim

WORKDIR /root

# Everything installed at build time — the task runs with no internet.
# numpy for the agent's convenience; pytest + ctrf plugin so the verifier
# can run from inside the image without downloading anything.
RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5

# Seed data only. generate_data.py (which holds the true drift constants)
# is deliberately NOT copied into the image.
COPY data/ /root/data/
