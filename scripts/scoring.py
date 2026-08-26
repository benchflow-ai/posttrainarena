"""PostTrain Arena — A3 scoring math (Δ vs θ_ref, bootstrap CIs, ≤20% domain cap).

The competition's scoring substrate, as a standalone, dependency-free, tested
module. Track 2 scores each team by the *improvement its submitted corpus buys*
on a held-out evaluation suite:

    Δ = PassRate_eval(θ_team) − PassRate_eval(θ_ref)

where PassRate is the mean per-instance verifier reward over the held-out suite
(BenchFlow Signals / IndexBench: 100 private + 20 public), and θ_ref is the fixed
reference checkpoint trained by the identical managed recipe on a fixed generic
corpus (the Δ *denominator* / control). Both checkpoints run the identical
rollout+verifier harness, so Δ isolates the team's contribution.

Two pieces this module adds over the raw held-out scorer:

  1. Bootstrap confidence intervals on Δ — a single pass-rate difference over
     ~120 instances is noisy; a paired instance-level bootstrap gives a CI and a
     "is Δ significantly > 0?" verdict, so the leaderboard reflects real signal,
     not sampling luck.

  2. The ≤20% per-domain cap (anti-hill-climbing) — without it a team can win by
     submitting a corpus that only lifts ONE eval domain. The cap guarantees no
     single domain contributes more than `cap` (default 20%) of a team's score,
     forcing breadth. Implemented by *iterative water-filling*, NOT naive
     clamp-then-renormalize: renormalizing after a clamp can push a clamped
     domain's weight back above the cap, silently violating the guarantee. Water-
     filling clamps the over-cap domains, redistributes the freed mass among the
     rest proportionally, and repeats until every domain weight ≤ cap.

Input contract (one team):

    {
      "team": "team-slug",
      "model_team": "...", "model_ref": "...",          # optional provenance
      "instances": [
        {"instance_id": "s2-code-0007", "domain": "code",
         "reward_ref": 0.0, "reward_team": 1.0},         # rewards in [0,1]
        ...
      ]
    }

Rewards may be binary (0/1 pass) or continuous verifier rewards in [0,1]; the
math (means, differences) is identical. Each instance must be scored by BOTH
checkpoints (paired) — that pairing is what makes the bootstrap valid.

CLI:

    python scripts/scoring.py --input scored_team.json
    python scripts/scoring.py --input scored_team.json --cap 0.2 --n-boot 10000 --seed 0
    python scripts/scoring.py --from-heldout env-heldout.json --domain code   # adapter
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InstanceScore:
    """One held-out eval instance, scored by both θ_team and θ_ref (paired)."""
    instance_id: str
    domain: str
    reward_ref: float
    reward_team: float


def index_instance_records(
    records: list[dict],
    *,
    label: str,
    id_key: str = "id",
) -> dict[str, dict]:
    """Strictly index per-instance records by id.

    Pairing is a data-integrity boundary: an empty result or a duplicate id must
    not be silently converted into a smaller (or overwritten) evaluation set.
    """
    if not records:
        raise ValueError(f"{label} has no per-instance records")

    indexed: dict[str, dict] = {}
    for pos, record in enumerate(records):
        raw_id = record.get(id_key)
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(f"{label} record {pos} has no {id_key!r}")
        iid = str(raw_id)
        if iid in indexed:
            raise ValueError(f"{label} has duplicate instance id {iid!r}")
        indexed[iid] = record
    return indexed


def pair_instance_records(
    ref_records: list[dict],
    team_records: list[dict],
    *,
    ref_label: str = "reference",
    team_label: str = "team",
    id_key: str = "id",
    domain_key: str = "domain",
    reward_key: str = "reward",
    default_domain: Optional[str] = None,
    instance_id_prefix: str = "",
) -> list[InstanceScore]:
    """Build strict paired scores from two per-instance result sets.

    Both arms must be non-empty, contain exactly the same unique ids, and agree
    on every instance's domain. This deliberately rejects partial intersections:
    scoring a conveniently overlapping subset can bias both the point estimate
    and its paired bootstrap confidence interval.
    """
    ref_by = index_instance_records(ref_records, label=ref_label, id_key=id_key)
    team_by = index_instance_records(team_records, label=team_label, id_key=id_key)

    ref_ids = set(ref_by)
    team_ids = set(team_by)
    if ref_ids != team_ids:
        missing_team = sorted(ref_ids - team_ids)
        missing_ref = sorted(team_ids - ref_ids)
        raise ValueError(
            f"instance id mismatch between {ref_label} and {team_label}: "
            f"missing from {team_label}={missing_team}; "
            f"missing from {ref_label}={missing_ref}")

    paired: list[InstanceScore] = []
    for iid in sorted(ref_ids):
        ref_record = ref_by[iid]
        team_record = team_by[iid]
        ref_domain_raw = ref_record.get(domain_key, default_domain)
        team_domain_raw = team_record.get(domain_key, default_domain)
        if ref_domain_raw is None or not str(ref_domain_raw).strip():
            raise ValueError(f"{ref_label} instance {iid!r} has no domain")
        if team_domain_raw is None or not str(team_domain_raw).strip():
            raise ValueError(f"{team_label} instance {iid!r} has no domain")
        ref_domain = str(ref_domain_raw)
        team_domain = str(team_domain_raw)
        if ref_domain != team_domain:
            raise ValueError(
                f"domain mismatch for instance {iid!r}: "
                f"{ref_label}={ref_domain!r}, {team_label}={team_domain!r}")
        if reward_key not in ref_record:
            raise ValueError(
                f"{ref_label} instance {iid!r} has no {reward_key!r}")
        if reward_key not in team_record:
            raise ValueError(
                f"{team_label} instance {iid!r} has no {reward_key!r}")
        ref_reward = float(ref_record[reward_key])
        team_reward = float(team_record[reward_key])
        for label, reward in ((ref_label, ref_reward), (team_label, team_reward)):
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError(
                    f"{label} instance {iid!r} has reward outside [0, 1]: "
                    f"{reward!r}")
        paired.append(InstanceScore(
            instance_id=f"{instance_id_prefix}{iid}",
            domain=ref_domain,
            reward_ref=ref_reward,
            reward_team=team_reward,
        ))
    return paired


def require_matching_result_provenance(
    ref_result: dict,
    team_result: dict,
    *,
    ref_label: str = "reference",
    team_label: str = "team",
    allow_legacy: bool = False,
) -> None:
    """Require two persisted evals to describe the exact same task/decode run.

    Eval IDs intentionally repeat across seeds, so equal ID sets alone cannot
    prove that two artifacts scored the same prompts. New artifacts persist both
    a canonical task-manifest hash and the generation/eval configuration. Legacy
    files can be admitted only through an explicit opt-in.
    """
    required = ("task_manifest_sha256", "eval_config")
    missing = {
        ref_label: [key for key in required if key not in ref_result],
        team_label: [key for key in required if key not in team_result],
    }
    if any(missing.values()):
        if not allow_legacy:
            raise ValueError(
                "cannot verify eval provenance; pass the explicit legacy override "
                f"only for trusted historical artifacts: {missing}")
    if ("task_manifest_sha256" in ref_result and
            "task_manifest_sha256" in team_result and
            ref_result["task_manifest_sha256"] != team_result["task_manifest_sha256"]):
        raise ValueError(
            f"task-manifest mismatch between {ref_label} and {team_label}")
    if ("eval_config" in ref_result and "eval_config" in team_result and
            ref_result["eval_config"] != team_result["eval_config"]):
        raise ValueError(
            f"eval-config mismatch between {ref_label} and {team_label}: "
            f"{ref_result['eval_config']!r} != {team_result['eval_config']!r}")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pass_rate(rewards: list[float]) -> float:
    """Mean verifier reward — the PassRate the competition reports."""
    return _mean(rewards)


# --------------------------------------------------------------------------- #
# Δ and its per-domain decomposition
# --------------------------------------------------------------------------- #
def overall_delta(instances: list[InstanceScore]) -> float:
    """Headline Δ = PassRate(θ_team) − PassRate(θ_ref) over the whole suite."""
    if not instances:
        return 0.0
    return (_mean([i.reward_team for i in instances])
            - _mean([i.reward_ref for i in instances]))


def per_domain(instances: list[InstanceScore]) -> dict[str, dict]:
    """Per-domain table: n, share, ref/team pass rate, and within-domain Δ_d.

    The identity  Δ_overall = Σ_d share_d · Δ_d  holds exactly (share_d = n_d/N),
    which is what the cap reweights.
    """
    by: dict[str, list[InstanceScore]] = {}
    for i in instances:
        by.setdefault(i.domain, []).append(i)
    n = len(instances)
    out: dict[str, dict] = {}
    for dom, items in by.items():
        ref = _mean([i.reward_ref for i in items])
        team = _mean([i.reward_team for i in items])
        out[dom] = {
            "n": len(items),
            "share": len(items) / n if n else 0.0,
            "ref_pass": ref,
            "team_pass": team,
            "delta": team - ref,
        }
    return out


# --------------------------------------------------------------------------- #
# The ≤cap per-domain weighting (iterative water-filling)
# --------------------------------------------------------------------------- #
def capped_weights(shares: dict[str, float], cap: float) -> dict[str, float]:
    """Weights that sum to 1 with EVERY weight ≤ cap, closest to `shares`.

    Iterative water-filling: clamp domains whose proportional weight exceeds the
    cap, freeze them at `cap`, redistribute the remaining mass among the rest in
    proportion to their original shares, and repeat until nothing exceeds the cap.

    Strictly enforces the anti-hill-climbing guarantee (no domain > cap of the
    total). Naive clamp-then-renormalize does NOT — renormalizing the survivors
    inflates a just-clamped domain back over the cap.

    Raises if the cap is infeasible (cap * n_domains < 1 — too few domains for
    every weight to fit under the cap).
    """
    keys = list(shares)
    if not keys:
        return {}
    k = len(keys)
    if cap * k < 1.0 - 1e-9:
        raise ValueError(
            f"cap={cap} infeasible for {k} domain(s): need cap >= 1/{k} = {1/k:.4f}")

    total = sum(shares.values()) or 1.0
    base = {key: shares[key] / total for key in keys}  # normalized natural shares

    clamped: dict[str, float] = {}
    free = set(keys)
    # At most k iterations: each pass clamps >=1 domain or terminates.
    for _ in range(k + 1):
        remaining_mass = 1.0 - sum(clamped.values())
        free_share = sum(base[key] for key in free) or 1.0
        # provisional proportional weights for the free set
        prov = {key: remaining_mass * (base[key] / free_share) for key in free}
        over = [key for key in free if prov[key] > cap + 1e-12]
        if not over:
            result = dict(clamped)
            result.update(prov)
            return result
        for key in over:
            clamped[key] = cap
            free.discard(key)
    # Fallback (numerical): clamp remainder uniformly. Should not be reached.
    result = dict(clamped)
    if free:
        rem = (1.0 - sum(clamped.values())) / len(free)
        for key in free:
            result[key] = rem
    return result


def capped_score(instances: list[InstanceScore], cap: float = 0.20) -> dict:
    """Team score with the ≤cap per-domain weighting applied.

    score = Σ_d w_d · Δ_d, where w_d are the water-filled weights (each ≤ cap).
    Also returns the per-domain contribution w_d·Δ_d and its share of the
    (positive) score, so the cap's effect is auditable.
    """
    dom = per_domain(instances)
    if not dom:
        return {"score": 0.0, "domains": {}, "cap": cap, "cap_applied": False}
    shares = {d: dom[d]["share"] for d in dom}
    # The cap is a multi-domain anti-hill-climbing rule. With fewer than 1/cap
    # domains it cannot bind (no set of weights ≤ cap sums to 1), so we fall back
    # to natural share weighting (score == overall Δ) and flag that the cap was
    # inert — better than crashing on a single-env or few-domain eval suite.
    try:
        w = capped_weights(shares, cap)
        cap_applied = True
    except ValueError:
        w = shares
        cap_applied = False
    score = sum(w[d] * dom[d]["delta"] for d in dom)
    # `capped_weight` (≤ cap) is the anti-hill-climbing guarantee; `contribution`
    # is the absolute w_d·Δ_d this domain adds to the score. `clamped` flags
    # domains whose natural share was pulled down to the cap.
    domains = {
        d: {
            **dom[d],
            "capped_weight": w[d],
            "contribution": w[d] * dom[d]["delta"],
            "clamped": dom[d]["share"] > w[d] + 1e-12,
        }
        for d in dom
    }
    return {"score": score, "cap": cap, "cap_applied": cap_applied, "domains": domains}


# --------------------------------------------------------------------------- #
# Paired instance-level bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    instances: list[InstanceScore],
    statistic: Callable[[list[InstanceScore]], float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for any instance-level statistic.

    Resamples the N instances with replacement (paired — each instance carries
    both its ref and team reward), recomputes `statistic`, and returns the
    [alpha/2, 1-alpha/2] percentile interval plus the point estimate and the
    fraction of resamples with statistic > 0 (a one-sided significance read).
    Deterministic given `seed`.
    """
    point = statistic(instances)
    n = len(instances)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 0,
                "alpha": alpha, "p_gt_0": 0.0, "significant": False}

    rng = random.Random(seed)
    idx = range(n)
    samples: list[float] = []
    n_pos = 0
    for _ in range(n_boot):
        pick = [instances[rng.choice(idx)] for _ in idx]
        s = statistic(pick)
        samples.append(s)
        if s > 0:
            n_pos += 1
    samples.sort()

    def _pct(p: float) -> float:
        if not samples:
            return 0.0
        k = min(len(samples) - 1, max(0, int(round(p * (len(samples) - 1)))))
        return samples[k]

    lo = _pct(alpha / 2)
    hi = _pct(1 - alpha / 2)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_boot": n_boot,
        "alpha": alpha,
        "p_gt_0": n_pos / n_boot,
        "significant": lo > 0,   # CI strictly above zero
    }


# --------------------------------------------------------------------------- #
# Top-level team report
# --------------------------------------------------------------------------- #
def score_team(
    instances: list[InstanceScore],
    cap: float = 0.20,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    team: Optional[str] = None,
) -> dict:
    """Full scoring report for one team: Δ, capped score, both with CIs."""
    if not instances:
        raise ValueError("cannot score an empty instance set")
    seen_ids: set[str] = set()
    for item in instances:
        if not item.instance_id or item.instance_id in seen_ids:
            raise ValueError(
                f"instance ids must be non-empty and unique; got {item.instance_id!r}")
        seen_ids.add(item.instance_id)
        if not item.domain:
            raise ValueError(f"instance {item.instance_id!r} has no domain")
        for name, reward in (("reference", item.reward_ref), ("team", item.reward_team)):
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError(
                    f"instance {item.instance_id!r} has {name} reward outside "
                    f"[0, 1]: {reward!r}")
    delta_ci = bootstrap_ci(instances, overall_delta, n_boot, alpha, seed)
    capped = capped_score(instances, cap)
    capped_ci = bootstrap_ci(
        instances, lambda xs: capped_score(xs, cap)["score"], n_boot, alpha, seed)

    return {
        "team": team,
        "n_instances": len(instances),
        "n_domains": len({i.domain for i in instances}),
        "ref_pass_rate": _mean([i.reward_ref for i in instances]),
        "team_pass_rate": _mean([i.reward_team for i in instances]),
        "overall_delta": delta_ci["point"],
        "overall_delta_ci": [delta_ci["lo"], delta_ci["hi"]],
        "overall_delta_significant": delta_ci["significant"],
        "overall_delta_p_gt_0": delta_ci["p_gt_0"],
        "capped_score": capped["score"],
        "capped_score_ci": [capped_ci["lo"], capped_ci["hi"]],
        "capped_score_significant": capped_ci["significant"],
        "cap": cap,
        "cap_applied": capped["cap_applied"],
        "per_domain": capped["domains"],
        "bootstrap": {"n_boot": n_boot, "alpha": alpha, "seed": seed},
    }


# --------------------------------------------------------------------------- #
# IO / adapters
# --------------------------------------------------------------------------- #
def load_instances(obj: dict) -> list[InstanceScore]:
    """Parse the canonical input contract into InstanceScore records."""
    out = []
    for r in obj.get("instances", []):
        out.append(InstanceScore(
            instance_id=str(r.get("instance_id", r.get("id", "?"))),
            domain=str(r.get("domain", "default")),
            reward_ref=float(r["reward_ref"]),
            reward_team=float(r["reward_team"]),
        ))
    return out


def from_heldout_json(path: str, domain: str = "default",
                      domain_key: str = "domain") -> list[InstanceScore]:
    """Adapter: build paired InstanceScores from a `<env>-heldout.json` file.

    run_heldout.py emits `per_instance` records tagged with a model label
    ("base"/"trained") and a split. We pair base(=ref) vs trained(=team) by
    instance id over the held-out split. If records lack a domain field, every
    instance is tagged `domain` (a single-env held-out file is one domain).
    """
    obj = json.loads(open(path, encoding="utf-8").read())
    per = obj.get("per_instance", [])
    ref_records: list[dict] = []
    team_records: list[dict] = []
    for pos, r in enumerate(per):
        split = str(r.get("phase", r.get("split", r.get("tag", ""))))
        split_lower = split.lower()
        if "heldout" not in split_lower and "held-out" not in split_lower:
            continue
        raw_id = r.get("inst_id", r.get("instance_id", r.get("id")))
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(f"held-out record {pos} has no instance id")
        iid = str(raw_id)
        reward = float(r.get("reward", r.get("score", 1.0 if r.get("passed") else 0.0)))
        record = {"id": iid, "domain": r.get(domain_key, domain), "reward": reward}

        arm_tokens = set(re.split(r"[^a-z0-9]+", split_lower))
        is_ref = bool(arm_tokens & {"base", "ref", "reference"})
        is_team = bool(arm_tokens & {"trained", "team"})
        if is_ref == is_team:
            raise ValueError(
                f"held-out record {iid!r} has ambiguous or unknown arm {split!r}")
        (ref_records if is_ref else team_records).append(record)

    return pair_instance_records(
        ref_records,
        team_records,
        ref_label="held-out reference arm",
        team_label="held-out trained arm",
    )


def _round(obj, nd=4):
    """Recursively round floats for clean JSON output."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PostTrain Arena A3 scoring math")
    ap.add_argument("--input", help="canonical team-scored JSON")
    ap.add_argument("--from-heldout", help="adapt a <env>-heldout.json instead")
    ap.add_argument("--domain", default="default",
                    help="domain tag for --from-heldout records lacking one")
    ap.add_argument("--cap", type=float, default=0.20, help="per-domain cap (default 0.20)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", help="write the full report JSON here")
    args = ap.parse_args(argv)

    team = None
    if args.from_heldout:
        instances = from_heldout_json(args.from_heldout, args.domain)
    elif args.input:
        obj = json.loads(open(args.input, encoding="utf-8").read())
        instances = load_instances(obj)
        team = obj.get("team")
    else:
        ap.error("one of --input or --from-heldout is required")
        return 2

    report = score_team(instances, cap=args.cap, n_boot=args.n_boot,
                        alpha=args.alpha, seed=args.seed, team=team)
    report = _round(report)
    text = json.dumps(report, indent=2)
    if args.output:
        open(args.output, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
