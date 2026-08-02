import pytest

pytest.importorskip("stable_baselines3")

from scripts.training._common import MilestoneGateCallback


def _cb() -> MilestoneGateCallback:
    return MilestoneGateCallback(milestones=[], window_rollouts=15, verbose=0)


def test_evaluate_checks_ok_fail_skip():
    cb = _cb()
    cb._recent["reward/place"].append(0.8)
    cb._recent["reward/place"].append(0.6)
    checks = [
        ("reward/place", 0.5, 0.0, "mean 0.7 >= 0.5 -> OK"),
        ("reward/place", 0.9, 0.0, "mean 0.7 < 0.9 -> FAIL"),
        ("metrics/unseen", 0.1, 0.0, "never recorded -> SKIP"),
    ]
    rows, failures = cb.evaluate_checks(checks)
    assert [r[0] for r in rows] == ["OK", "FAIL", "SKIP"]
    assert [f[0] for f in failures] == ["reward/place"]


def test_evaluate_checks_no_failure_when_all_above_floor():
    cb = _cb()
    cb._recent["metrics/insertion_depth"].append(0.05)
    cb._recent["reward/place_release"].append(0.9)
    checks = [
        ("metrics/insertion_depth", 0.001, 0.0, "insertion exists"),
        ("reward/place_release", 0.5, 0.0, "engaged-release paid"),
    ]
    rows, failures = cb.evaluate_checks(checks)
    assert failures == []
    assert all(r[0] == "OK" for r in rows)


def test_evaluate_checks_flags_never_inserts():
    cb = _cb()
    cb._recent["metrics/insertion_depth"].append(0.0)
    _, failures = cb.evaluate_checks(
        [("metrics/insertion_depth", 0.001, 0.0, "must insert at all")]
    )
    assert [f[0] for f in failures] == ["metrics/insertion_depth"]


def _peg_10m_checks():
    from scripts.training.train_peg import PEG_GATES

    step, checks, _ = PEG_GATES[0]
    assert step == 10_000_000
    return checks


def test_peg_10m_gate_kills_a_collapsed_run():
    cb = _cb()
    for key, value in (("reward/place", 0.0109), ("metrics/stage", 0.98)):
        cb._recent[key].append(value)

    _, failures = cb.evaluate_checks(_peg_10m_checks())
    assert {f[0] for f in failures} == {"reward/place", "metrics/stage"}


def test_peg_10m_gate_passes_run7_which_was_killed():
    cb = _cb()
    for key, value in (("reward/place", 0.4947), ("metrics/stage", 2.1276)):
        cb._recent[key].append(value)

    _, failures = cb.evaluate_checks(_peg_10m_checks())
    assert failures == [], f"run 7 would be killed again by {[f[0] for f in failures]}"


def _peg_gate_checks(step):
    from scripts.training.train_peg import PEG_GATES

    return next(checks for s, checks, _ in PEG_GATES if s == step)


def test_peg_30m_gate_passes_run8_which_was_killed():
    cb = _cb()
    for key, value in (("reward/place", 0.6861), ("metrics/insertion_depth", 0.0300)):
        cb._recent[key].append(value)

    _, failures = cb.evaluate_checks(_peg_gate_checks(30_000_000))
    assert failures == [], f"run 8 would be killed again by {[f[0] for f in failures]}"


def test_no_gate_uses_a_metric_that_collapses_on_success():
    from scripts.training.train_peg import PEG_GATES

    banned = {
        "metrics/num_finger_contacts",
        "reward/lift",
        "reward/grasp",
        "reward/holding",
        "reward/axis_in_grip",
        "metrics/peg_height",
    }

    for step, checks, _ in PEG_GATES:
        used = {c[0] for c in checks}
        assert not (used & banned), (
            f"{step:,} gate uses {used & banned} — anti-correlated with success"
        )


def _gate_checks(gates, step):
    return next(checks for s, checks, _ in gates if s == step)


def test_pickplace_gates_pass_the_run_that_solved_it():
    from scripts.training.train_pickplace import PICKPLACE_GATES

    for step, measured in (
        (10_000_000, {"metrics/num_finger_contacts": 1.3313, "reward/lifting": 0.4764}),
        (30_000_000, {"reward/placed": 0.6054, "reward/success": 3.6653}),
    ):
        cb = _cb()
        for key, value in measured.items():
            cb._recent[key].append(value)
        _, failures = cb.evaluate_checks(_gate_checks(PICKPLACE_GATES, step))
        assert failures == [], (
            f"{step:,} would kill the verified run via {[f[0] for f in failures]}"
        )


def test_pickplace_30m_gate_uses_no_transient_carry_metric():
    from scripts.training.train_pickplace import PICKPLACE_GATES

    banned = {
        "reward/transport",
        "reward/lifting",
        "reward/grasping",
        "reward/grasp_quality",
        "reward/holding",
        "metrics/num_finger_contacts",
    }

    for step, checks, _ in PICKPLACE_GATES:
        if step <= 10_000_000:
            continue
        used = {c[0] for c in checks}
        assert not (used & banned), f"{step:,} gate uses {used & banned} — collapses on success"


def test_grasp_gates_pass_the_run_that_solved_it():
    from scripts.training.train_grasp import GRASP_GATES

    for step, measured in (
        (10_000_000, {"metrics/num_finger_contacts": 1.7051, "reward/grasping": 0.5436}),
        (30_000_000, {"reward/lifting": 0.8455, "metrics/success_hold_steps": 54.3834}),
    ):
        cb = _cb()
        for key, value in measured.items():
            cb._recent[key].append(value)
        _, failures = cb.evaluate_checks(_gate_checks(GRASP_GATES, step))
        assert failures == [], (
            f"{step:,} would kill the verified run via {[f[0] for f in failures]}"
        )


def test_floor_adaptive_lr_bounds_the_sbx_minibatch_walk():
    KLAdaptiveLR = pytest.importorskip("sbx.common.utils").KLAdaptiveLR
    from scripts.training._common import LR_FLOOR, LR_KL_MARGIN, floor_adaptive_lr

    class _Model:
        pass

    model = _Model()
    model.adaptive_lr = KLAdaptiveLR(0.05, 3e-4)
    assert model.adaptive_lr.min_learning_rate == 1e-5, "sbx default changed — re-derive"
    assert model.adaptive_lr.kl_margin == 2.0, "sbx default changed — re-derive"

    floor_adaptive_lr(model)
    assert model.adaptive_lr.min_learning_rate == LR_FLOOR
    assert model.adaptive_lr.kl_margin == LR_KL_MARGIN

    floor_adaptive_lr(_Model())


def test_lr_deadband_covers_the_observed_kl_range():
    from scripts.training._common import LR_KL_MARGIN

    target_kl = 0.05
    assert target_kl / LR_KL_MARGIN < 0.040, "lower edge would still raise the lr"
    assert target_kl * LR_KL_MARGIN > 0.141, "upper edge would still cut the lr"


def test_peg_10m_gate_does_not_demand_terminal_metrics():
    keys = {c[0] for c in _peg_10m_checks()}
    for terminal in ("reward/place_release", "reward/success", "metrics/insertion_depth"):
        assert terminal not in keys, (
            f"{terminal} is a terminal metric and must not gate 10M — the policy is "
            f"still mid-carry there"
        )
