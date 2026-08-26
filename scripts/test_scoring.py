"""Tests for scripts/scoring.py (A3 scoring math).

Runs standalone (`python scripts/test_scoring.py`) or under pytest. No third-party
deps. Each test asserts a concrete property of Δ, the ≤cap water-filling, the
bootstrap CI, or the held-out adapter.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scoring as S
from scoring import InstanceScore as I


def _assert_value_error(fn, message_fragment):
    try:
        fn()
        assert False, f"expected ValueError containing {message_fragment!r}"
    except ValueError as exc:
        assert message_fragment in str(exc), str(exc)


# --------------------------------------------------------------------------- #
# Δ and pass rate
# --------------------------------------------------------------------------- #
def test_pass_rate_and_delta():
    insts = [
        I("a", "code", 0.0, 1.0),
        I("b", "code", 1.0, 1.0),
        I("c", "math", 0.0, 0.0),
        I("d", "math", 0.0, 1.0),
    ]
    assert S.pass_rate([i.reward_ref for i in insts]) == 0.25
    assert S.pass_rate([i.reward_team for i in insts]) == 0.75
    assert abs(S.overall_delta(insts) - 0.5) < 1e-12


def test_delta_empty_is_zero():
    assert S.overall_delta([]) == 0.0
    assert S.pass_rate([]) == 0.0


def test_continuous_rewards():
    # rewards need not be binary; means still work
    insts = [I("a", "d", 0.2, 0.8), I("b", "d", 0.4, 0.6)]
    assert abs(S.overall_delta(insts) - 0.4) < 1e-12


# --------------------------------------------------------------------------- #
# Strict per-instance indexing + pairing
# --------------------------------------------------------------------------- #
def test_strict_index_rejects_empty_and_duplicate_ids():
    _assert_value_error(
        lambda: S.index_instance_records([], label="reference"),
        "no per-instance records",
    )
    duplicate = [
        {"id": "i1", "domain": "code", "reward": 0.0},
        {"id": "i1", "domain": "code", "reward": 1.0},
    ]
    _assert_value_error(
        lambda: S.index_instance_records(duplicate, label="reference"),
        "duplicate instance id",
    )


def test_strict_pairing_preserves_valid_behavior_and_prefix():
    ref = [
        {"id": "i2", "domain": "math", "reward": 1.0},
        {"id": "i1", "domain": "code", "reward": 0.0},
    ]
    team = [
        {"id": "i1", "domain": "code", "reward": 1.0},
        {"id": "i2", "domain": "math", "reward": 0.0},
    ]
    paired = S.pair_instance_records(ref, team, instance_id_prefix="s0-")
    assert paired == [
        I("s0-i1", "code", 0.0, 1.0),
        I("s0-i2", "math", 1.0, 0.0),
    ]


def test_strict_pairing_rejects_mismatched_ids_and_domains():
    ref = [{"id": "i1", "domain": "code", "reward": 0.0}]
    wrong_id = [{"id": "i2", "domain": "code", "reward": 1.0}]
    _assert_value_error(
        lambda: S.pair_instance_records(ref, wrong_id),
        "instance id mismatch",
    )
    wrong_domain = [{"id": "i1", "domain": "math", "reward": 1.0}]
    _assert_value_error(
        lambda: S.pair_instance_records(ref, wrong_domain),
        "domain mismatch",
    )
    bad_reward = [{"id": "i1", "domain": "code", "reward": 1.1}]
    _assert_value_error(
        lambda: S.pair_instance_records(ref, bad_reward),
        "reward outside [0, 1]",
    )


def test_result_provenance_requires_exact_manifest_and_config():
    ref = {
        "task_manifest_sha256": "abc123",
        "eval_config": {"seed": 0, "difficulty": "expert"},
    }
    team = {
        "task_manifest_sha256": "abc123",
        "eval_config": {"seed": 0, "difficulty": "expert"},
    }
    S.require_matching_result_provenance(ref, team)

    bad_manifest = {**team, "task_manifest_sha256": "different"}
    _assert_value_error(
        lambda: S.require_matching_result_provenance(ref, bad_manifest),
        "task-manifest mismatch",
    )
    bad_config = {**team, "eval_config": {"seed": 1, "difficulty": "expert"}}
    _assert_value_error(
        lambda: S.require_matching_result_provenance(ref, bad_config),
        "eval-config mismatch",
    )
    _assert_value_error(
        lambda: S.require_matching_result_provenance({}, {}),
        "cannot verify eval provenance",
    )
    S.require_matching_result_provenance({}, {}, allow_legacy=True)


# --------------------------------------------------------------------------- #
# Per-domain decomposition identity
# --------------------------------------------------------------------------- #
def test_per_domain_decomposition_identity():
    # Δ_overall must equal Σ_d share_d · Δ_d exactly.
    insts = [
        I("a", "code", 0.0, 1.0), I("b", "code", 0.0, 1.0), I("c", "code", 1.0, 1.0),
        I("d", "math", 0.0, 0.0), I("e", "math", 0.0, 1.0),
        I("f", "stem", 1.0, 1.0),
    ]
    dom = S.per_domain(insts)
    recomposed = sum(dom[d]["share"] * dom[d]["delta"] for d in dom)
    assert abs(recomposed - S.overall_delta(insts)) < 1e-12
    assert dom["code"]["n"] == 3
    assert abs(sum(dom[d]["share"] for d in dom) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# ≤cap water-filling — the load-bearing anti-hill-climbing guarantee
# --------------------------------------------------------------------------- #
def test_capped_weights_sum_to_one_and_respect_cap():
    shares = {"a": 0.60, "b": 0.10, "c": 0.10, "d": 0.10, "e": 0.10}
    w = S.capped_weights(shares, cap=0.20)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    for k, v in w.items():
        assert v <= 0.20 + 1e-9, f"{k} weight {v} exceeds cap"


def test_water_filling_beats_naive_renormalize():
    # The case where naive clamp-then-renormalize FAILS:
    #   clamp a=0.60->0.20, leave b..e=0.10; sum=0.60; renormalize -> a=0.333 (>cap!)
    # Water-filling must keep a at exactly the cap.
    shares = {"a": 0.60, "b": 0.10, "c": 0.10, "d": 0.10, "e": 0.10}
    w = S.capped_weights(shares, cap=0.20)
    assert abs(w["a"] - 0.20) < 1e-9, f"clamped domain should sit at cap, got {w['a']}"
    # freed mass (0.80) split evenly across the four 0.10-share domains -> 0.20 each
    for k in ("b", "c", "d", "e"):
        assert abs(w[k] - 0.20) < 1e-9


def test_capped_weights_uniform_passthrough():
    # already-balanced shares are unchanged
    shares = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    w = S.capped_weights(shares, cap=0.30)
    for k in shares:
        assert abs(w[k] - 0.25) < 1e-9


def test_capped_weights_infeasible_raises():
    # 3 domains, cap 0.20 -> max total 0.60 < 1.0 -> infeasible (primitive is strict)
    try:
        S.capped_weights({"a": 0.5, "b": 0.3, "c": 0.2}, cap=0.20)
        assert False, "expected ValueError for infeasible cap"
    except ValueError:
        pass


def test_capped_score_falls_back_when_cap_infeasible():
    # single domain: cap can't bind -> graceful fallback to natural weighting,
    # score == overall Δ, flagged cap_applied=False (must NOT raise).
    insts = [I(f"x{i}", "code", 0.0, 1.0 if i < 5 else 0.0) for i in range(10)]
    cs = S.capped_score(insts, cap=0.20)
    assert cs["cap_applied"] is False
    assert abs(cs["score"] - S.overall_delta(insts)) < 1e-12
    # and the full report survives a single-domain suite end to end
    rep = S.score_team(insts, cap=0.20, n_boot=200, seed=0)
    assert rep["cap_applied"] is False
    assert rep["n_domains"] == 1


def test_two_pass_water_filling():
    # one giant + one medium domain both need clamping across iterations
    shares = {"a": 0.50, "b": 0.30, "c": 0.06, "d": 0.06, "e": 0.04, "f": 0.04}
    w = S.capped_weights(shares, cap=0.20)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    for k, v in w.items():
        assert v <= 0.20 + 1e-9


# --------------------------------------------------------------------------- #
# Capped score caps a hill-climber's reward
# --------------------------------------------------------------------------- #
def test_hill_climber_is_capped():
    # Team A: a huge lift concentrated in ONE domain (60% of instances), nothing
    # elsewhere. Team B: a modest, even lift across five domains. The cap should
    # stop A from dominating purely by piling into one domain.
    big = ([I(f"a{i}", "code", 0.0, 1.0) for i in range(60)]
           + [I(f"a{i}", d, 0.0, 0.0)
              for d, base in (("math", 60), ("stem", 70), ("tool", 80), ("chat", 90))
              for i in range(base, base + 10)])
    a_uncapped = S.overall_delta(big)
    a_capped = S.capped_score(big, cap=0.20)["score"]
    # uncapped Δ ~ 0.6; capped score must be much smaller (one domain clamped to 20%)
    assert a_uncapped > 0.55
    assert a_capped < a_uncapped
    assert a_capped <= 0.20 + 1e-9   # a single positive domain can contribute <= cap

    # the real guarantee: the over-represented domain's WEIGHT is clamped to cap
    cs = S.capped_score(big, cap=0.20)
    assert cs["domains"]["code"]["capped_weight"] <= 0.20 + 1e-9
    assert cs["domains"]["code"]["clamped"] is True
    # and its absolute score contribution is therefore bounded by cap·Δ_d
    assert cs["domains"]["code"]["contribution"] <= 0.20 + 1e-9


def test_even_team_beats_concentrated_under_cap():
    # five domains, even +0.2 lift each
    even = []
    for d in ("code", "math", "stem", "tool", "chat"):
        for i in range(20):
            team = 1.0 if i < 4 else 0.0   # +0.2 per domain
            even.append(I(f"{d}{i}", d, 0.0, team))
    # one domain +1.0, rest flat
    conc = []
    for j, d in enumerate(("code", "math", "stem", "tool", "chat")):
        for i in range(20):
            team = 1.0 if d == "code" else 0.0
            conc.append(I(f"{d}{i}", d, 0.0, team))
    even_score = S.capped_score(even, cap=0.20)["score"]
    conc_score = S.capped_score(conc, cap=0.20)["score"]
    assert even_score > conc_score, (even_score, conc_score)


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #
def test_bootstrap_deterministic():
    insts = [I(f"x{i}", "d", 0.0, 1.0 if i % 2 else 0.0) for i in range(40)]
    a = S.bootstrap_ci(insts, S.overall_delta, n_boot=500, seed=7)
    b = S.bootstrap_ci(insts, S.overall_delta, n_boot=500, seed=7)
    assert a == b   # same seed -> identical (the real contract)
    # the seed must actually drive the RNG: across several seeds, the CI endpoints
    # are not all identical (any *single* pair may coincide on a coarse statistic).
    cis = {(S.bootstrap_ci(insts, S.overall_delta, n_boot=500, seed=s)["lo"],
            S.bootstrap_ci(insts, S.overall_delta, n_boot=500, seed=s)["hi"])
           for s in range(10)}
    assert len(cis) > 1, "seed has no effect on the bootstrap draw"


def test_bootstrap_brackets_point_and_significance():
    # strong, unambiguous lift: every instance goes 0 -> 1
    insts = [I(f"x{i}", "d", 0.0, 1.0) for i in range(50)]
    ci = S.bootstrap_ci(insts, S.overall_delta, n_boot=1000, seed=0)
    assert ci["point"] == 1.0
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["significant"] and ci["lo"] > 0
    assert ci["p_gt_0"] == 1.0


def test_bootstrap_noise_not_significant():
    # team identical to ref -> Δ=0, CI must straddle 0, not "significant"
    insts = [I(f"x{i}", "d", 1.0 if i % 2 else 0.0, 1.0 if i % 2 else 0.0)
             for i in range(60)]
    ci = S.bootstrap_ci(insts, S.overall_delta, n_boot=1000, seed=1)
    assert ci["point"] == 0.0
    assert not ci["significant"]
    assert ci["lo"] <= 0.0 <= ci["hi"]


def test_bootstrap_ci_narrows_with_more_data():
    def width(n):
        insts = [I(f"x{i}", "d", 0.0, 1.0 if i % 3 == 0 else 0.0) for i in range(n)]
        ci = S.bootstrap_ci(insts, S.overall_delta, n_boot=800, seed=3)
        return ci["hi"] - ci["lo"]
    assert width(200) < width(20)


# --------------------------------------------------------------------------- #
# Full report + adapter
# --------------------------------------------------------------------------- #
def test_score_team_smoke():
    insts = []
    for d in ("code", "math", "stem", "tool", "chat"):
        for i in range(24):
            insts.append(I(f"{d}{i}", d, 0.0, 1.0 if i < 6 else 0.0))
    rep = S.score_team(insts, cap=0.20, n_boot=400, seed=0, team="demo")
    assert rep["team"] == "demo"
    assert rep["n_domains"] == 5
    assert rep["n_instances"] == 120
    assert abs(rep["overall_delta"] - 0.25) < 1e-9
    assert len(rep["overall_delta_ci"]) == 2
    assert "code" in rep["per_domain"]


def test_score_team_rejects_empty_duplicate_and_out_of_range_inputs():
    _assert_value_error(lambda: S.score_team([]), "empty instance set")
    duplicate = [I("same", "code", 0.0, 1.0), I("same", "math", 0.0, 1.0)]
    _assert_value_error(lambda: S.score_team(duplicate), "unique")
    _assert_value_error(
        lambda: S.score_team([I("bad", "code", 0.0, float("nan"))]),
        "outside [0, 1]",
    )


def test_from_heldout_adapter_pairs_base_and_trained():
    payload = {
        "env": "demo-env",
        "per_instance": [   # real run_heldout.py schema: `phase` field
            {"inst_id": "i1", "phase": "base-heldout", "reward": 0.0},
            {"inst_id": "i1", "phase": "trained-heldout", "reward": 1.0},
            {"inst_id": "i2", "phase": "base-heldout", "reward": 0.0},
            {"inst_id": "i2", "phase": "trained-heldout", "reward": 1.0},
            # train-split rows must be ignored
            {"inst_id": "i9", "phase": "base-train", "reward": 1.0},
            {"inst_id": "i9", "phase": "trained-train", "reward": 1.0},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        insts = S.from_heldout_json(path, domain="agentic")
        assert len(insts) == 2                       # only held-out, paired
        assert {i.instance_id for i in insts} == {"i1", "i2"}
        assert all(i.domain == "agentic" for i in insts)
        assert abs(S.overall_delta(insts) - 1.0) < 1e-12
    finally:
        os.unlink(path)


def _assert_heldout_error(per_instance, message_fragment):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"per_instance": per_instance}, f)
        path = f.name
    try:
        _assert_value_error(
            lambda: S.from_heldout_json(path, domain="agentic"),
            message_fragment,
        )
    finally:
        os.unlink(path)


def test_from_heldout_rejects_duplicate_and_mismatched_arms():
    _assert_heldout_error([
        {"inst_id": "i1", "phase": "base-heldout", "reward": 0.0},
        {"inst_id": "i1", "phase": "base-heldout", "reward": 0.5},
        {"inst_id": "i1", "phase": "trained-heldout", "reward": 1.0},
    ], "duplicate instance id")

    _assert_heldout_error([
        {"inst_id": "i1", "phase": "base-heldout", "reward": 0.0},
        {"inst_id": "i2", "phase": "trained-heldout", "reward": 1.0},
    ], "instance id mismatch")


def test_from_heldout_rejects_domain_and_unknown_arm_mismatch():
    _assert_heldout_error([
        {"inst_id": "i1", "phase": "base-heldout", "domain": "code", "reward": 0.0},
        {"inst_id": "i1", "phase": "trained-heldout", "domain": "math", "reward": 1.0},
    ], "domain mismatch")

    _assert_heldout_error([
        {"inst_id": "i1", "phase": "mystery-heldout", "reward": 0.0},
    ], "ambiguous or unknown arm")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
