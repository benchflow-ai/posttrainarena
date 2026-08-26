#!/usr/bin/env python3
"""Auto-validate and attribute a model + data mixture on an eval mixture.

This is the PostTrain Arena attribution pipeline. Given a run spec it answers
one question end to end:

    does THIS model (optionally trained on THIS data mixture) beat a baseline
    on THIS eval mixture, and by how much — with a confidence interval and a
    per-domain breakdown that no single domain can dominate?

It is a thin orchestrator over BenchFlow (>= 0.6.7). BenchFlow already runs any
task source against any ACP agent in any sandbox (`bench eval run`) and computes
a paired lift report with bootstrap CIs between two job directories
(`bench eval compare-lift`). What it does not do is span an *eval mixture* —
several task sources scored as one suite — or apply the competition's
anti-hill-climbing cap. That is what this adds:

    1. fan out `bench eval run` over (arm x eval-mixture component)
    2. per-component paired lift via BenchFlow's compare-lift
    3. aggregate components into one attributed score, with the <=20%
       per-domain cap from scripts/scoring.py

Everything else — sandboxing, verifier hardening, trajectory capture, rollout
selection, bootstrap resampling — is BenchFlow's and is deliberately not
reimplemented here.

Usage:
    python3 scripts/pta_attribute.py --spec configs/attribution/lhtb-27b.yaml
    python3 scripts/pta_attribute.py --spec ... --stage eval      # runs only
    python3 scripts/pta_attribute.py --spec ... --stage attribute # score only
    python3 scripts/pta_attribute.py --spec ... --dry-run         # print plan
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import capped_weights  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in the docs
    print("pta_attribute: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    raise

ARMS = ("baseline", "candidate")
DEFAULT_CAP = 0.20
DEFAULT_BOOTSTRAP = 1000

# Subcommands this pipeline depends on. `compare-lift` landed after 0.6.0, and a
# stale CLI fails with a bare "No such command", which is easy to misread as a
# bug here rather than a wrong binary.
REQUIRED_BENCH_COMMANDS = ("compare-lift",)


def resolve_bench() -> str:
    """Resolve the BenchFlow CLI, preferring an explicit override.

    PATH order differs between an interactive shell and a subprocess: a pyenv
    shim can shadow the uv tool install, so `bench` in a terminal and `bench`
    from Python are not necessarily the same binary or the same version. Always
    resolve once, explicitly, and pass the resolved path to subprocess.
    """
    import os

    override = os.environ.get("BENCH_BIN")
    if override:
        return override
    return shutil.which("bench") or "bench"


def check_agents(bench: str, spec: Spec) -> str | None:
    """Verify every arm's agent is registered with BenchFlow.

    Agent names are NOT portable between harnesses: another harness's agent
    name is not a BenchFlow one. An unregistered name does not fail fast — the
    run provisions a sandbox and only then dies with
    "ACP initialize timed out after 60s", which looks like a network or image
    problem rather than a typo. Catch it before spending.
    """
    listing = subprocess.run(
        [bench, "agent", "list"], capture_output=True, text=True, timeout=120
    )
    if listing.returncode != 0:
        return None  # cannot enumerate; let the run proceed rather than block it

    known: set[str] = set()
    for line in listing.stdout.splitlines():
        for cell in line.split("│"):
            token = cell.strip()
            if token and all(c.isalnum() or c in "-_" for c in token):
                known.add(token)
    if not known:
        return None

    bad = sorted(
        {
            arm.agent
            for arm in spec.arms.values()
            # Registered names may be truncated in the table, so accept a prefix
            # match against any known token.
            if not any(k.startswith(arm.agent) or arm.agent.startswith(k) for k in known)
        }
    )
    if bad:
        return (
            f"agent(s) {', '.join(bad)} are not registered with BenchFlow.\n"
            "Agent names are not portable between harnesses.\n"
            f"Run `{bench} agent list` for the registered set (openclaw takes any "
            "--model at runtime and needs no agent-specific key)."
        )
    return None


def check_bench(bench: str) -> str | None:
    """Return an error message if `bench` is unusable, else None."""
    try:
        version = subprocess.run(
            [bench, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not execute {bench!r}: {exc}"
    if version.returncode != 0:
        return f"{bench} --version failed: {version.stderr.strip()[:200]}"

    for command in REQUIRED_BENCH_COMMANDS:
        probe = subprocess.run(
            [bench, "eval", command, "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if probe.returncode != 0:
            return (
                f"{bench} ({version.stdout.strip()}) has no `bench eval {command}`.\n"
                "This is almost always a stale CLI shadowing the one you installed "
                "— a pyenv shim ahead of ~/.local/bin is the usual cause.\n"
                "Fix with:\n"
                "  uv tool install --python 3.12 --prerelease allow --upgrade benchflow\n"
                "then point this run at it explicitly:\n"
                "  BENCH_BIN=$(ls ~/.local/bin/bench) python3 scripts/pta_attribute.py ..."
            )
    return None


# --------------------------------------------------------------------------- #
# Run spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalComponent:
    """One task source inside an eval mixture.

    `domain` is the attribution bucket the component contributes to. Several
    components may share a domain (e.g. two SWE sources both scoring as
    "software-engineering"), which is what the per-domain cap acts on.
    """

    name: str
    domain: str
    source_repo: str | None = None
    source_path: str | None = None
    source_ref: str | None = None
    tasks_dir: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    expected_tasks: int | None = None
    weight: float = 1.0
    config_override: str | None = None
    # Local checkout used only for preflight inspection when the tasks are
    # fetched from a remote source_repo at run time.
    local_mirror: str | None = None
    # Materialize a sanitized copy of the tasks (unsupported harness keys
    # stripped) and run THAT instead. Requires tasks_dir or local_mirror.
    sanitize: bool = False

    def preflight_dir(self) -> Path | None:
        path = self.tasks_dir or self.local_mirror
        return Path(path) if path else None

    def source_args(self, sanitized_dir: Path | None = None) -> list[str]:
        args: list[str] = []
        if sanitized_dir is not None:
            # A sanitized tree replaces the remote source entirely.
            args += ["--tasks-dir", str(sanitized_dir)]
        else:
            if self.tasks_dir:
                args += ["--tasks-dir", self.tasks_dir]
            if self.source_repo:
                args += ["--source-repo", self.source_repo]
            if self.source_path:
                args += ["--source-path", self.source_path]
            if self.source_ref:
                args += ["--source-ref", self.source_ref]
        for name in self.include:
            args += ["--include", name]
        for name in self.exclude:
            args += ["--exclude", name]
        if self.expected_tasks is not None:
            args += ["--expected-tasks", str(self.expected_tasks)]
        if self.config_override:
            args += ["--config-override", self.config_override]
        return args


# Harness features a task may declare that the BenchFlow runner does not
# implement.
#
# These do NOT degrade gracefully. BenchFlow's TaskConfig is
# `ConfigDict(extra="forbid")`, so an unknown key under `[agent]` fails task
# loading outright:
#
#   ValidationError: 1 validation error for TaskConfig
#   agent.continue_until_timeout
#     Extra inputs are not permitted [type=extra_forbidden]
#
# `continue_until_timeout` is LHTB's, set on 30 of its 46 tasks: the agent is
# meant to keep working (with interim verifier feedback) until its timeout
# rather than stopping at first completion. Left in place, every one of those
# tasks errors instead of running.
#
# `sanitize_tasks()` strips the key so the task loads. That is a real semantic
# change — the task then runs single-shot and scores below its reference
# harness — so it is recorded as a limitation rather than done silently.
UNSUPPORTED_HARNESS_FEATURES = ("continue_until_timeout",)


# Root-level keys BenchFlow parses but cannot execute on any sandbox. These are
# rejected by `validate_task_runtime_support` rather than by pydantic, so they
# survive the extra-forbid check and fail later with
# "Task uses parsed runtime features that BenchFlow cannot execute".
#
# `artifacts` is LHTB's declared output-file list. It is rejected on BOTH
# daytona and docker ("root artifact collection is parsed but not
# runtime-gated"), so it is a BenchFlow gap, not a sandbox choice. Stripping it
# means those files are not copied into the trial directory; it does NOT change
# scoring, because the verifier runs inside the sandbox against the live
# filesystem. The cost is post-hoc inspection of outputs, not reward validity.
UNSUPPORTED_ROOT_KEYS = ("artifacts",)


def _strip_keys(text: str, keys: tuple[str, ...]) -> tuple[str, set[str]]:
    """Drop `key = ...` assignments, including multi-line array values."""
    kept: list[str] = []
    removed: set[str] = set()
    in_array = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if in_array:
            if stripped.endswith("]"):
                in_array = False
            continue
        head = stripped.split("=", 1)[0].strip()
        if "=" in stripped and head in keys:
            removed.add(head)
            value = stripped.split("=", 1)[1].strip()
            # An array value may continue over following lines.
            if value.startswith("[") and not value.endswith("]"):
                in_array = True
            continue
        kept.append(line)
    return "".join(kept), removed


def sanitize_tasks(src: Path, dst: Path) -> dict[str, int]:
    """Copy a task tree with keys BenchFlow cannot execute removed.

    Returns {key: tasks_stripped}. The copy is what gets run; the original tree
    is never modified.

    Two independent gates are handled, because a task can fail either way:
      * `UNSUPPORTED_HARNESS_FEATURES` under `[agent]` -> pydantic extra-forbid
      * `UNSUPPORTED_ROOT_KEYS` at the root -> runtime-capability rejection
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    keys = UNSUPPORTED_HARNESS_FEATURES + UNSUPPORTED_ROOT_KEYS
    stripped: dict[str, int] = {}
    for toml_path in sorted(dst.glob("*/task.toml")):
        try:
            text = toml_path.read_text(encoding="utf-8")
        except OSError:
            continue
        new_text, removed = _strip_keys(text, keys)
        if removed:
            toml_path.write_text(new_text, encoding="utf-8")
            for key in removed:
                stripped[key] = stripped.get(key, 0) + 1
    return stripped


def scan_harness_gaps(tasks_dir: Path) -> dict[str, int]:
    """Count tasks declaring harness features the runner will ignore."""
    counts: dict[str, int] = {}
    if not tasks_dir.is_dir():
        return counts
    for toml_path in sorted(tasks_dir.glob("*/task.toml")):
        try:
            text = toml_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for feature in UNSUPPORTED_HARNESS_FEATURES:
            # Cheap textual probe: avoids a tomllib dependency on 3.10 and is
            # good enough for a warning (false positives only over-warn).
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(feature) and "true" in stripped.lower():
                    counts[feature] = counts.get(feature, 0) + 1
                    break
    return counts


@dataclass(frozen=True)
class Arm:
    """One side of the paired comparison."""

    name: str
    agent: str
    model: str
    reasoning_effort: str | None = None
    api_base: str | None = None


@dataclass(frozen=True)
class Spec:
    name: str
    arms: dict[str, Arm]
    components: tuple[EvalComponent, ...]
    sandbox: str = "daytona"
    concurrency: int = 100
    worker_concurrency: int | None = None
    jobs_root: Path = Path("jobs")
    n_attempts: int = 1
    cap: float = DEFAULT_CAP
    bootstrap_samples: int = DEFAULT_BOOTSTRAP
    bootstrap_seed: int = 0
    data_mixture: tuple[dict[str, Any], ...] = ()
    extra_eval_args: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


def _expand_env(arg: str) -> str:
    # Specs carry no secrets; keys arrive via ${VAR} references resolved here.
    # An unresolved reference means the operator forgot to export the var, and
    # passing it through literally just breaks auth three layers down — so die.
    expanded = os.path.expandvars(arg)
    if "${" in expanded:
        sys.exit(f"pta_attribute: unresolved env reference in extra_eval_args: {arg}")
    return expanded


def load_spec(path: Path) -> Spec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    arms_raw = raw.get("arms") or {}
    missing = [a for a in ARMS if a not in arms_raw]
    if missing:
        raise SystemExit(
            f"spec {path}: arms must define both 'baseline' and 'candidate' "
            f"(missing: {', '.join(missing)})"
        )
    arms = {
        name: Arm(
            name=name,
            agent=str(arms_raw[name]["agent"]),
            model=str(arms_raw[name]["model"]),
            reasoning_effort=arms_raw[name].get("reasoning_effort"),
            api_base=arms_raw[name].get("api_base"),
        )
        for name in ARMS
    }

    mixture = raw.get("eval_mixture") or []
    if not mixture:
        raise SystemExit(f"spec {path}: eval_mixture must list at least one component")

    components = []
    for entry in mixture:
        if not entry.get("name"):
            raise SystemExit(f"spec {path}: every eval_mixture component needs a name")
        if not (entry.get("source_repo") or entry.get("tasks_dir")):
            raise SystemExit(
                f"spec {path}: component {entry['name']} needs source_repo or tasks_dir"
            )
        components.append(
            EvalComponent(
                name=str(entry["name"]),
                domain=str(entry.get("domain") or entry["name"]),
                source_repo=entry.get("source_repo"),
                source_path=entry.get("source_path"),
                source_ref=entry.get("source_ref"),
                tasks_dir=entry.get("tasks_dir"),
                # YAML coerces bare numeric task names (e.g. the LHTB task
                # `2048`) to int, which breaks argv construction — force str.
                include=tuple(str(x) for x in (entry.get("include") or ())),
                exclude=tuple(str(x) for x in (entry.get("exclude") or ())),
                expected_tasks=entry.get("expected_tasks"),
                weight=float(entry.get("weight", 1.0)),
                config_override=entry.get("config_override"),
                local_mirror=entry.get("local_mirror"),
                sanitize=bool(entry.get("sanitize", False)),
            )
        )

    attribution = raw.get("attribution") or {}
    return Spec(
        name=str(raw.get("name") or path.stem),
        arms=arms,
        components=tuple(components),
        sandbox=str(raw.get("sandbox") or "daytona"),
        concurrency=int(raw.get("concurrency", 100)),
        worker_concurrency=raw.get("worker_concurrency"),
        jobs_root=Path(raw.get("jobs_root") or "jobs"),
        n_attempts=int(raw.get("n_attempts", 1)),
        cap=float(attribution.get("per_domain_cap", DEFAULT_CAP)),
        bootstrap_samples=int(attribution.get("bootstrap_samples", DEFAULT_BOOTSTRAP)),
        bootstrap_seed=int(attribution.get("bootstrap_seed", 0)),
        data_mixture=tuple(raw.get("data_mixture") or ()),
        extra_eval_args=tuple(_expand_env(a) for a in (raw.get("extra_eval_args") or ())),
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Stage 1 — fan out evals
# --------------------------------------------------------------------------- #


def jobs_dir_for(spec: Spec, arm: str, component: EvalComponent) -> Path:
    return spec.jobs_root / spec.name / arm / component.name


def eval_command(
    spec: Spec,
    arm: Arm,
    component: EvalComponent,
    bench: str = "bench",
    sanitized_dir: Path | None = None,
) -> list[str]:
    cmd = [
        bench,
        "eval",
        "run",
        "--agent",
        arm.agent,
        "--model",
        arm.model,
        "--sandbox",
        spec.sandbox,
        "--concurrency",
        str(spec.concurrency),
        "--jobs-dir",
        str(jobs_dir_for(spec, arm.name, component)),
        "--task-manifest-out",
        str(jobs_dir_for(spec, arm.name, component) / "task-manifest.json"),
    ]
    if arm.reasoning_effort:
        cmd += ["--reasoning-effort", arm.reasoning_effort]
    if spec.worker_concurrency:
        cmd += ["--worker-concurrency", str(spec.worker_concurrency)]
    if spec.n_attempts != 1:
        cmd += ["--n-attempts", str(spec.n_attempts)]
    cmd += component.source_args(sanitized_dir)
    cmd += list(spec.extra_eval_args)
    return cmd


def preflight(spec: Spec) -> dict[str, dict[str, int]]:
    """Warn about harness features the runner will ignore, before spending.

    Returns {component_name: {feature: task_count}} for anything found.
    """
    found: dict[str, dict[str, int]] = {}
    for component in spec.components:
        directory = component.preflight_dir()
        if directory is None:
            continue
        gaps = scan_harness_gaps(directory)
        if gaps:
            found[component.name] = gaps
            for feature, count in sorted(gaps.items()):
                print(
                    f"[preflight] {component.name}: {count} task(s) declare "
                    f"`{feature}`, which BenchFlow cannot execute — those tasks "
                    "FAIL TO LOAD unless the component sets `sanitize: true`. "
                    "Sanitizing runs them in a reduced regime, so absolute scores "
                    "are NOT comparable to published numbers for this suite.",
                    file=sys.stderr,
                )
    return found


def run_evals(
    spec: Spec,
    *,
    dry_run: bool,
    only_arm: str | None = None,
    bench: str = "bench",
    sanitized: dict[str, Path] | None = None,
) -> None:
    sanitized = sanitized or {}
    for arm_name in ARMS:
        if only_arm and arm_name != only_arm:
            continue
        arm = spec.arms[arm_name]
        for component in spec.components:
            cmd = eval_command(spec, arm, component, bench, sanitized.get(component.name))
            print(f"[eval] {arm_name}/{component.name}: {' '.join(cmd)}", flush=True)
            if dry_run:
                continue
            out = jobs_dir_for(spec, arm_name, component)
            out.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                # A component failing is recoverable: attribution reports coverage
                # and excludes it rather than pretending the suite completed.
                print(
                    f"[eval] WARNING {arm_name}/{component.name} exited "
                    f"{proc.returncode}; it will show as missing coverage",
                    file=sys.stderr,
                )


# --------------------------------------------------------------------------- #
# Stage 2 — per-component lift via BenchFlow
# --------------------------------------------------------------------------- #


def component_lift(
    spec: Spec, component: EvalComponent, bench: str = "bench"
) -> dict[str, Any] | None:
    """Run `bench eval compare-lift` for one component, return its report."""

    baseline = jobs_dir_for(spec, "baseline", component)
    candidate = jobs_dir_for(spec, "candidate", component)
    if not baseline.exists() or not candidate.exists():
        print(
            f"[lift] skipping {component.name}: missing job dir "
            f"(baseline={baseline.exists()} candidate={candidate.exists()})",
            file=sys.stderr,
        )
        return None

    report_path = spec.jobs_root / spec.name / "lift" / f"{component.name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        bench,
        "eval",
        "compare-lift",
        "--baseline",
        str(baseline),
        "--trained",
        str(candidate),
        "--bootstrap-samples",
        str(spec.bootstrap_samples),
        "--bootstrap-seed",
        str(spec.bootstrap_seed),
        "--json-out",
        str(report_path),
        "--out",
        str(report_path.with_suffix(".md")),
    ]
    print(f"[lift] {component.name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 or not report_path.exists():
        print(f"[lift] WARNING compare-lift failed for {component.name}", file=sys.stderr)
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


# BenchFlow's lift report (schema_version 1) exposes both a binary pass-rate
# delta and a dense mean-reward delta. We attribute on MEAN REWARD: LHTB-style
# suites are floor-bound under a binary criterion (29 of 46 tasks are solved by
# no model at all), so pass-rate deltas collapse to zero while mean reward still
# separates models. pass_rate_delta is carried along for reference only.
LIFT_METRIC = "mean_reward_delta"


def _extract_delta(report: dict[str, Any], metric: str = LIFT_METRIC) -> float | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _extract_ci(report: dict[str, Any], metric: str = LIFT_METRIC) -> list[float] | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    ci = metrics.get("ci")
    if not isinstance(ci, dict):
        return None
    value = ci.get(metric)
    if isinstance(value, dict):
        lo, hi = value.get("low"), value.get("high")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return [float(lo), float(hi)]
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [float(value[0]), float(value[1])]
    return None


def _paired_count(report: dict[str, Any]) -> int | None:
    metrics = report.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("paired_count"), int):
        return metrics["paired_count"]
    pairing = report.get("pairing")
    if isinstance(pairing, dict):
        value = pairing.get("paired_tasks_count")
        if isinstance(value, int):
            return value
    return None


# --------------------------------------------------------------------------- #
# Stage 3 — aggregate with the per-domain cap
# --------------------------------------------------------------------------- #


def attribute(
    spec: Spec,
    reports: dict[str, dict[str, Any]],
    harness_gaps: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Combine per-component lift into one capped, attributed score."""

    harness_gaps = harness_gaps or {}

    per_domain: dict[str, dict[str, Any]] = {}
    for component in spec.components:
        report = reports.get(component.name)
        if report is None:
            continue
        delta = _extract_delta(report)
        if delta is None:
            print(
                f"[attribute] WARNING no delta found in lift report for "
                f"{component.name}; excluded from the score",
                file=sys.stderr,
            )
            continue
        bucket = per_domain.setdefault(
            component.domain, {"components": [], "weight_raw": 0.0, "deltas": []}
        )
        bucket["components"].append(
            {
                "name": component.name,
                "delta": delta,
                "ci": _extract_ci(report),
                "pass_rate_delta": _extract_delta(report, "pass_rate_delta"),
                "paired_count": _paired_count(report),
                "weight": component.weight,
                "lift_report": str(
                    spec.jobs_root / spec.name / "lift" / f"{component.name}.json"
                ),
            }
        )
        bucket["weight_raw"] += component.weight
        bucket["deltas"].append((component.weight, delta))

    if not per_domain:
        return {
            "spec": spec.name,
            "generated_at": datetime.now(UTC).isoformat(),
            "score": 0.0,
            "domains": {},
            "cap": spec.cap,
            "cap_applied": False,
            "coverage": {"components_scored": 0, "components_total": len(spec.components)},
            "limitations": ["No component produced a usable lift report."],
        }

    # Within a domain, components combine by their own weights.
    for bucket in per_domain.values():
        total_w = sum(w for w, _ in bucket["deltas"]) or 1.0
        bucket["delta"] = sum(w * d for w, d in bucket["deltas"]) / total_w
        del bucket["deltas"]

    total_raw = sum(b["weight_raw"] for b in per_domain.values()) or 1.0
    shares = {d: b["weight_raw"] / total_raw for d, b in per_domain.items()}

    cap_applied = False
    if len(shares) * spec.cap >= 1.0:
        weights = capped_weights(shares, spec.cap)
        cap_applied = any(
            abs(weights[d] - shares[d]) > 1e-9 for d in shares
        )
    else:
        # Too few domains for the cap to be satisfiable; report honestly rather
        # than silently dropping the anti-hill-climbing guarantee.
        weights = shares

    score = sum(weights[d] * per_domain[d]["delta"] for d in per_domain)

    for domain, bucket in per_domain.items():
        bucket["share_raw"] = shares[domain]
        bucket["weight"] = weights[domain]
        bucket["contribution"] = weights[domain] * bucket["delta"]

    limitations = []
    scored = sum(len(b["components"]) for b in per_domain.values())
    if scored < len(spec.components):
        limitations.append(
            f"Only {scored}/{len(spec.components)} eval-mixture components produced "
            "a lift report; the score covers the scored subset only."
        )
    if len(shares) * spec.cap < 1.0:
        limitations.append(
            f"The <={spec.cap:.0%} per-domain cap needs at least "
            f"{int(1 / spec.cap + 0.999)} domains to bind; this mixture has "
            f"{len(shares)}, so the cap was NOT applied and a single domain can "
            "dominate the score."
        )
    for name, gaps in sorted(harness_gaps.items()):
        for feature, count in sorted(gaps.items()):
            limitations.append(
                f"Component '{name}': {count} task(s) declare `{feature}`, which "
                "the BenchFlow runner does not support (its TaskConfig forbids "
                "unknown keys, so the task fails to load). Sanitizing strips the "
                "key and the task runs in a reduced regime instead. The paired Δ "
                "stays internally valid (both arms run the same regime) but "
                "absolute scores are NOT comparable to externally published "
                "numbers for this suite, and the reduced regime narrows the "
                "discriminating band."
            )

    return {
        "spec": spec.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "score": score,
        "cap": spec.cap,
        "cap_applied": cap_applied,
        "bootstrap_samples": spec.bootstrap_samples,
        "metric": LIFT_METRIC,
        "harness_gaps": harness_gaps,
        "domains": per_domain,
        "coverage": {
            "components_scored": scored,
            "components_total": len(spec.components),
            "domains": len(per_domain),
        },
        "arms": {
            name: {"agent": arm.agent, "model": arm.model} for name, arm in spec.arms.items()
        },
        "data_mixture": list(spec.data_mixture),
        "limitations": limitations,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Attribution — {report['spec']}",
        "",
        f"**Score (capped Δ): {report['score']:+.4f}**  ",
        f"generated {report['generated_at']} · "
        f"cap {report['cap']:.0%} "
        f"({'applied' if report.get('cap_applied') else 'not binding'}) · "
        f"{report.get('bootstrap_samples')} bootstrap samples",
        "",
        "| domain | weight | Δ | contribution | components |",
        "|---|---:|---:|---:|---|",
    ]
    for domain, bucket in sorted(
        report.get("domains", {}).items(),
        key=lambda kv: -abs(kv[1].get("contribution", 0.0)),
    ):
        names = ", ".join(c["name"] for c in bucket["components"])
        lines.append(
            f"| {domain} | {bucket['weight']:.3f} | {bucket['delta']:+.4f} | "
            f"{bucket['contribution']:+.4f} | {names} |"
        )
    cov = report.get("coverage", {})
    lines += [
        "",
        f"Coverage: {cov.get('components_scored')}/{cov.get('components_total')} "
        f"components across {cov.get('domains')} domains.",
    ]
    arms = report.get("arms", {})
    if arms:
        lines += ["", "| arm | agent | model |", "|---|---|---|"]
        for name, arm in arms.items():
            lines.append(f"| {name} | {arm['agent']} | `{arm['model']}` |")
    if report.get("limitations"):
        lines += ["", "## Limitations", ""]
        lines += [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, type=Path, help="run-spec YAML")
    ap.add_argument(
        "--stage",
        choices=("all", "eval", "attribute"),
        default="all",
        help="run everything, only the evals, or only the scoring",
    )
    ap.add_argument("--arm", choices=ARMS, help="restrict --stage eval to one arm")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--output", type=Path, help="write the attribution JSON here")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    harness_gaps = preflight(spec)
    bench = resolve_bench()

    # Materialize sanitized task trees for components that ask for it. Without
    # this, a task carrying an unsupported harness key does not degrade — it
    # fails to load, and the whole component reports as missing coverage.
    sanitized: dict[str, Path] = {}
    for component in spec.components:
        if not component.sanitize:
            continue
        source = component.preflight_dir()
        if source is None or not source.is_dir():
            print(
                f"pta_attribute: component {component.name} sets sanitize: true "
                "but has no local tasks_dir/local_mirror to sanitize from",
                file=sys.stderr,
            )
            return 2
        target = spec.jobs_root / spec.name / "_sanitized" / component.name
        if not args.dry_run:
            stripped = sanitize_tasks(source, target)
            if stripped:
                for feature, count in sorted(stripped.items()):
                    print(
                        f"[sanitize] {component.name}: stripped `{feature}` from "
                        f"{count} task(s) -> {target}",
                        flush=True,
                    )
                harness_gaps.setdefault(component.name, {}).update(stripped)
        sanitized[component.name] = target

    # Verify the CLI once, up front, for any stage that shells out to it.
    if not args.dry_run:
        problem = check_bench(bench)
        if problem:
            print(f"pta_attribute: {problem}", file=sys.stderr)
            return 2
        if args.stage in ("all", "eval"):
            agent_problem = check_agents(bench, spec)
            if agent_problem:
                print(f"pta_attribute: {agent_problem}", file=sys.stderr)
                return 2
        print(f"[bench] using {bench}", flush=True)

    if args.stage in ("all", "eval"):
        run_evals(
            spec,
            dry_run=args.dry_run,
            only_arm=args.arm,
            bench=bench,
            sanitized=sanitized,
        )

    if args.dry_run or args.stage == "eval":
        return 0

    reports: dict[str, dict[str, Any]] = {}
    for component in spec.components:
        report = component_lift(spec, component, bench)
        if report is not None:
            reports[component.name] = report

    attribution = attribute(spec, reports, harness_gaps)
    out = args.output or (spec.jobs_root / spec.name / "attribution.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(attribution, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(attribution), encoding="utf-8")

    print()
    print(render_markdown(attribution))
    print(f"[attribute] wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
