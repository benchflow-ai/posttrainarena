# Fireworks deployment safety

<!-- markdownlint-disable MD013 -->

This runbook prevents an evaluation deployment from continuing to consume
dedicated GPU capacity after the evaluation has stopped.

## What the deployment email means

The message

> Your deployment has been created and is now running.

is a billing notification, not only a readiness notification. An on-demand
deployment is billed per active GPU-second, independent of traffic. A stopped,
failed, or idle evaluation does not stop the deployment automatically unless
its scaling configuration permits zero replicas.

Before creating a deployment, check the current rates on the
[Fireworks pricing page](https://fireworks.ai/pricing). Estimate the hourly
cost as:

```text
GPU price per hour × accelerators per replica × active replicas
```

For example, on September 9, 2026, a replica using two B200 GPUs at
`$13/GPU-hour` cost `$26/hour`.

## Safe defaults for evaluation

Use these settings for temporary evaluation deployments:

- `minReplicaCount = 0`
- `maxReplicaCount = 1`, unless measured concurrency requires more
- scale-to-zero enabled
- preemptible capacity when interruption and retry are acceptable
- one named owner responsible for the post-run check

Setting a scale-to-zero window is insufficient when `minReplicaCount` is `1`.
The minimum replica count wins, so the deployment remains active and billable
without traffic.

Preemptible deployments are appropriate for retryable batch evaluation because
they do not reserve paid dedicated capacity while idle. They can be reclaimed
without warning and are not suitable for latency-sensitive production serving.
See the [on-demand deployment
guide](https://docs.fireworks.ai/guides/ondemand-deployments) for current
behavior and creation options.

## Preflight checklist

Before starting an evaluation:

1. Record the deployment ID, base model, GPU type, accelerators per replica,
   and current hourly price.
2. Confirm `minReplicaCount = 0`.
3. Confirm the maximum replica count matches the intended budget.
4. Calculate and record the maximum expected spend:

   ```text
   hourly cost × maximum planned runtime
   ```

5. Confirm the evaluation process has both `FIREWORKS_API_KEY` and
   `DAYTONA_API_KEY`.
6. Run one task before starting the complete matrix.
7. Verify the canary produced a healthy trajectory and did not fail because of
   a missing API key or an agent idle timeout.
8. Set a calendar or external alert for the planned stop time. Do not rely on
   the evaluation process to clean up a deployment after a crash or disconnect.

Never print, commit, or paste `.env` values into logs. Keep shell tracing
disabled while handling credentials:

```bash
set +x
set -a
source experiments/skillsbench-fireworks-rft/.env
set +a
```

## Inspect all deployments

Set the account ID explicitly:

```bash
export FIREWORKS_ACCOUNT_ID=xiangyi-0cb55c
```

List deployment state, scaling bounds, replicas, and GPU allocation:

```bash
curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  "https://api.fireworks.ai/v1/accounts/$FIREWORKS_ACCOUNT_ID/deployments" |
python3 -c '
import json
import sys

for item in json.load(sys.stdin).get("deployments", []):
    print(
        item["name"].rsplit("/", 1)[-1],
        "state=%s" % item.get("state"),
        "min=%s" % item.get("minReplicaCount"),
        "max=%s" % item.get("maxReplicaCount"),
        "replicas=%s" % item.get("replicaCount"),
        "gpu=%sx%s"
        % (item.get("acceleratorType"), item.get("acceleratorCount")),
    )
'
```

A deployment stops consuming active GPU time only once `replicaCount` reaches
`0`. An `UPDATING` response is not proof that billing has stopped.

## Emergency stop

Export the deployment identity and its exact base model:

```bash
export FIREWORKS_DEPLOYMENT_ID=sbds41-pass10
export FIREWORKS_BASE_MODEL=accounts/xiangyi-0cb55c/models/sbds41-r32-e10-50cac45a
export FIREWORKS_DEPLOYMENT_URL="https://api.fireworks.ai/v1/accounts/$FIREWORKS_ACCOUNT_ID/deployments/$FIREWORKS_DEPLOYMENT_ID"
```

First pin both scaling bounds to zero. The update request must include the base
model; omitting it can leave the previous replica bounds in effect.

```bash
curl --fail-with-body --silent --show-error \
  -X PATCH \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -H "Content-Type: application/json" \
  "$FIREWORKS_DEPLOYMENT_URL?updateMask=minReplicaCount,maxReplicaCount" \
  --data "{
    \"baseModel\": \"$FIREWORKS_BASE_MODEL\",
    \"minReplicaCount\": 0,
    \"maxReplicaCount\": 0
  }"
```

Then force the active replica count to zero:

```bash
curl --fail-with-body --silent --show-error \
  -X PATCH \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -H "Content-Type: application/json" \
  "$FIREWORKS_DEPLOYMENT_URL:scale" \
  --data '{"replicaCount": 0}'
```

Finally, poll the deployment until all three values are zero:

```bash
while true; do
  status="$(
    curl --fail-with-body --silent --show-error \
      -H "Authorization: Bearer $FIREWORKS_API_KEY" \
      "$FIREWORKS_DEPLOYMENT_URL"
  )"
  STATUS="$status" python3 -c '
import json
import os

item = json.loads(os.environ["STATUS"])
print(
    "state=%s" % item.get("state"),
    "min=%s" % item.get("minReplicaCount"),
    "max=%s" % item.get("maxReplicaCount"),
    "replicas=%s" % item.get("replicaCount"),
)
'
  bounds="$(
    STATUS="$status" python3 -c '
import json
import os

item = json.loads(os.environ["STATUS"])
print(
    item.get("minReplicaCount"),
    item.get("maxReplicaCount"),
    item.get("replicaCount"),
)
'
  )"
  [ "$bounds" = "0 0 0" ] && break
  sleep 15
done
```

Keeping `maxReplicaCount = 0` is an intentional hard stop. Requests cannot
scale the deployment back up until an operator raises the maximum.

## Re-arm a stopped deployment safely

To allow on-demand use again without pinning an always-on replica, set the
bounds to `min = 0` and `max = 1`:

```bash
curl --fail-with-body --silent --show-error \
  -X PATCH \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -H "Content-Type: application/json" \
  "$FIREWORKS_DEPLOYMENT_URL?updateMask=minReplicaCount,maxReplicaCount" \
  --data "{
    \"baseModel\": \"$FIREWORKS_BASE_MODEL\",
    \"minReplicaCount\": 0,
    \"maxReplicaCount\": 1
  }"
```

Traffic may then scale the deployment up. After the evaluation, repeat the
inspection and emergency-stop procedure rather than assuming scale-down has
completed.

## Post-run checklist

The operator who started the run owns this checklist:

1. Stop the evaluation process and confirm no retry worker remains active.
2. Set the deployment maximum and active replica count to zero.
3. Poll until `replicaCount = 0`.
4. List every deployment in the account and confirm none has active replicas.
5. Check training, batch inference, and evaluation jobs for nonterminal states.
6. Record start time, zero-replica time, GPU allocation, and estimated spend.
7. Preserve failure artifacts before deleting a deployment.

Scaling to zero preserves the deployment identity, but not necessarily
indefinitely: Fireworks documents automatic deletion of zero-minimum
deployments after extended inactivity. The model is a separate resource and is
not deleted merely because its deployment scales to zero or is removed.

## Repository-specific notes

The SkillsBench Fireworks evaluation reads its model and deployment from
`experiments/skillsbench-fireworks-rft/eval-matrix.yaml`. If a deployment is
deleted and recreated under a different ID, update that file before running:

```bash
python experiments/skillsbench-fireworks-rft/scripts/run_pass_at_k.py \
  --k 1 \
  --concurrency 1 \
  --jobs-dir /tmp/skillsbench-fireworks-canary
```

Only increase `--k` and concurrency after the canary completes with healthy
trajectories. A ready deployment does not imply that the evaluation client has
the correct credentials or that the agent is making progress.
