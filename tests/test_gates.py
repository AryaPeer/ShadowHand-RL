import pytest

pytest.importorskip("stable_baselines3")

from shadow_hand.training.callbacks import MilestoneGateCallback
from shadow_hand.training.gates import PEG_GATES


def _cb() -> MilestoneGateCallback:
    return MilestoneGateCallback(milestones=[], window_rollouts=15, verbose=0)


def test_evaluate_checks_ok_fail_skip():
    cb = _cb()
    cb._recent["reward/place"].append(0.8)
    cb._recent["reward/place"].append(0.6)
    checks = [
        ("reward/place", 0.5, "mean 0.7 >= 0.5 -> OK"),
        ("reward/place", 0.9, "mean 0.7 < 0.9 -> FAIL"),
        ("metrics/unseen", 0.1, "never recorded -> SKIP"),
    ]
    rows, failures = cb.evaluate_checks(checks)
    assert [r[0] for r in rows] == ["OK", "FAIL", "SKIP"]
    assert [f[0] for f in failures] == ["reward/place"]


def test_evaluate_checks_no_failure_when_all_above_floor():
    cb = _cb()
    cb._recent["metrics/insertion_depth"].append(0.05)
    cb._recent["reward/place_release"].append(0.9)
    checks = [
        ("metrics/insertion_depth", 0.001, "insertion exists"),
        ("reward/place_release", 0.5, "engaged-release paid"),
    ]
    rows, failures = cb.evaluate_checks(checks)
    assert failures == []
    assert all(r[0] == "OK" for r in rows)


def _peg_10m_checks():
    step, checks, _ = PEG_GATES[0]
    assert step == 10_000_000
    return checks


def test_peg_10m_gate_fails_below_floor():
    cb = _cb()
    for key, value in (("reward/place", 0.0109), ("metrics/stage", 0.98)):
        cb._recent[key].append(value)

    _, failures = cb.evaluate_checks(_peg_10m_checks())
    assert {f[0] for f in failures} == {"reward/place", "metrics/stage"}


def test_floor_adaptive_lr_raises_sbx_bounds():
    KLAdaptiveLR = pytest.importorskip("sbx.common.utils").KLAdaptiveLR
    from shadow_hand.training.runner import LR_FLOOR, LR_KL_MARGIN, floor_adaptive_lr

    baseline = "sbx defaults are the baseline these overrides raise"

    class _Model:
        pass

    model = _Model()
    model.adaptive_lr = KLAdaptiveLR(0.05, 3e-4)
    assert model.adaptive_lr.min_learning_rate == 1e-5, baseline
    assert model.adaptive_lr.kl_margin == 2.0, baseline

    floor_adaptive_lr(model)
    assert model.adaptive_lr.min_learning_rate == LR_FLOOR
    assert model.adaptive_lr.kl_margin == LR_KL_MARGIN

    floor_adaptive_lr(_Model())
