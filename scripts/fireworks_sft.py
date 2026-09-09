#!/usr/bin/env python3
"""Create a Fireworks dataset and launch a managed LoRA SFT job.

This is the managed fine-tuning path behind every checkpoint in
docs/sft-attribution-ledger.md, replacing the retired local Slurm route.
Rows are chat JSONL; only the `messages` field is uploaded, and --split
selects rows by their `split` field so a corpus with an embedded holdout
uploads its training rows only.

    FIREWORKS_API_KEY=fw_... python3 scripts/fireworks_sft.py \
        --data skillsbench_oracle_sft.jsonl --split train \
        --dataset-id sb-oracle-86 --output-model sbo86-r32-1e4 \
        --epochs 3 --lr 1e-4 --lora-rank 32

    python3 scripts/fireworks_sft.py --status \
        accounts/<account>/supervisedFineTuningJobs/<job-id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API_BASE = "https://api.fireworks.ai/v1"


def api(key: str, method: str, path: str, body: bytes | None = None,
        content_type: str = "application/json", tolerate: str | None = None) -> dict:
    """One REST call. `tolerate` skips an expected error (idempotent re-runs)."""
    req = urllib.request.Request(
        f"{API_BASE}/{path.lstrip('/')}",
        method=method,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": content_type,
            "User-Agent": "posttrain-arena-sft/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        if tolerate and tolerate in detail:
            print(f"{path}: {tolerate} -- continuing")
            return {}
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {detail}")


def resolve_account(key: str) -> str:
    accounts = api(key, "GET", "accounts")["accounts"]
    if len(accounts) != 1:
        raise SystemExit(f"expected one account, found {len(accounts)}; pass --account")
    return accounts[0]["name"].split("/")[-1]


def load_rows(path: Path, split: str | None) -> list[dict]:
    rows = []
    for line_num, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        rec = json.loads(line)
        if split and rec.get("split") != split:
            continue
        messages = rec.get("messages")
        if messages is None:
            prompt, completion = rec.get("prompt"), rec.get("completion")
            if not isinstance(prompt, list) or not isinstance(completion, list):
                raise SystemExit(
                    f"{path}:{line_num}: expected messages or prompt/completion"
                )
            messages = prompt + completion
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"{path}:{line_num}: messages must be a non-empty list")
        normalized = []
        for message in messages:
            clean = dict(message)
            for tool_call in clean.get("tool_calls") or []:
                function = tool_call.get("function")
                if isinstance(function, dict) and isinstance(
                    function.get("arguments"), dict
                ):
                    function["arguments"] = json.dumps(
                        function["arguments"],
                        separators=(",", ":"),
                    )
            normalized.append(clean)
        rows.append({"messages": normalized})
    if not rows:
        raise SystemExit(f"no rows selected from {path} (split={split!r})")
    return rows


def multipart(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; "
        f'name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/jsonl\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def create_dataset(key: str, account: str, dataset_id: str, rows: list[dict]) -> str:
    name = f"accounts/{account}/datasets/{dataset_id}"
    api(key, "POST", f"accounts/{account}/datasets", json.dumps({
        "datasetId": dataset_id,
        "dataset": {"userUploaded": {}, "exampleCount": str(len(rows))},
    }).encode(), tolerate="already exists")
    if api(key, "GET", name).get("state") != "READY":
        payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
        body, ctype = multipart("file", f"{dataset_id}.jsonl", payload)
        api(key, "POST", f"accounts/{account}/datasets/{dataset_id}:upload", body, ctype)
        api(key, "POST", f"accounts/{account}/datasets/{dataset_id}:validateUpload",
            b"{}", tolerate="already uploaded")
    for _ in range(60):
        state = api(key, "GET", name).get("state")
        if state == "READY":
            print(f"dataset {name}: READY ({len(rows)} rows)")
            return name
        if state not in ("UPLOADING", "DATASET_STATE_UNSPECIFIED"):
            raise SystemExit(f"dataset {name} entered state {state}")
        time.sleep(5)
    raise SystemExit(f"dataset {name} still not READY; check the Fireworks console")


def create_job(key: str, account: str, args: argparse.Namespace, dataset: str) -> dict:
    body = {
        "dataset": dataset,
        "baseModel": args.base_model,
        "outputModel": f"accounts/{account}/models/{args.output_model}",
        "displayName": args.display_name or args.output_model,
        "epochs": args.epochs,
        "learningRate": args.lr,
        "loraRank": args.lora_rank,
        "maxContextLength": args.max_context_length,
    }
    if args.batch_size_samples:
        body["batchSizeSamples"] = args.batch_size_samples
    if args.evaluation_dataset:
        body["evaluationDataset"] = f"accounts/{account}/datasets/{args.evaluation_dataset}"
    return api(key, "POST", f"accounts/{account}/supervisedFineTuningJobs",
               json.dumps(body).encode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", metavar="JOB_NAME",
                    help="print the state of an existing job and exit")
    ap.add_argument("--data", type=Path, help="chat JSONL to upload")
    ap.add_argument("--split", default=None,
                    help="only upload rows whose `split` field matches")
    ap.add_argument("--dataset-id", help="Fireworks dataset id to create")
    ap.add_argument("--eval-data", type=Path, default=None,
                    help="chat JSONL to upload as the validation set")
    ap.add_argument("--eval-split", default=None,
                    help="`split` filter applied to --eval-data")
    ap.add_argument("--evaluation-dataset", default=None,
                    help="Fireworks dataset id for validation (created from "
                         "--eval-data when that is given)")
    ap.add_argument("--account", default=None)
    ap.add_argument("--base-model", default="accounts/fireworks/models/qwen3p6-27b")
    ap.add_argument("--output-model", help="model id for the tuned adapter")
    ap.add_argument("--display-name", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-context-length", type=int, default=32768)
    ap.add_argument("--batch-size-samples", type=int, default=None)
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("set FIREWORKS_API_KEY", file=sys.stderr)
        return 2

    if args.status:
        job = api(key, "GET", args.status)
        print(json.dumps({k: job.get(k) for k in
                          ("name", "state", "status", "outputModel", "createTime",
                           "completedTime", "trainedTokens")}, indent=2))
        return 0

    if not (args.data and args.dataset_id and args.output_model):
        ap.error("--data, --dataset-id, and --output-model are required to launch")
    account = args.account or resolve_account(key)
    rows = load_rows(args.data, args.split)
    dataset = create_dataset(key, account, args.dataset_id, rows)
    if args.eval_data:
        if not args.evaluation_dataset:
            ap.error("--eval-data requires --evaluation-dataset")
        create_dataset(key, account, args.evaluation_dataset,
                       load_rows(args.eval_data, args.eval_split))
    job = create_job(key, account, args, dataset)
    print(f"job     {job['name']}")
    print(f"state   {job.get('state')}")
    print(f"output  {job.get('outputModel')}")
    print(f"monitor python3 scripts/fireworks_sft.py --status {job['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
