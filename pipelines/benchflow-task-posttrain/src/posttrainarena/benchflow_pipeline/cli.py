"""Command-line interface for the public BenchFlow pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrainarena-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in {"plan", "run"}:
            command.add_argument("--run-name")
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        print(
            json.dumps(
                {"config": str(config.source), "valid": True},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    pipeline = Pipeline(
        config,
        run_name=args.run_name,
        dry_run=getattr(args, "dry_run", False),
        resume=getattr(args, "resume", False),
    )
    if args.command == "plan":
        print(json.dumps(pipeline.plan(), indent=2, sort_keys=True, default=str))
        return 0
    result = pipeline.run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
