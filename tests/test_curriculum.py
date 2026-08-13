from unittest.mock import MagicMock

import pytest

from shadow_hand.config import AdaptiveCurriculumConfig
from shadow_hand.curriculum.callbacks import (
    AdaptiveCurriculumCallback,
)


def _setup_callback(cb) -> MagicMock:

    mock_env = MagicMock()
    cb.locals = {}
    cb.globals = {}
    cb.model = MagicMock()
    cb.model.get_env.return_value = mock_env
    cb.num_timesteps = 0
    return mock_env


_LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _make_adaptive(start_idx=0, **overrides):
    defaults = {
        "clearance": 0.010,
        "carry_floor_levels": _LEVELS,
        "advance_success_frac": 0.5,
        "retreat_success_frac": 0.15,
        "check_interval_rollouts": 3,
        "success_window_rollouts": 3,
    }
    defaults.update(overrides)
    cb = AdaptiveCurriculumCallback(
        AdaptiveCurriculumConfig(**defaults), success_bonus_per_step=5.0, verbose=0
    )
    cb._level_idx = start_idx
    cb._carry_floor = cb._levels[start_idx]
    mock_env = _setup_callback(cb)
    return cb, mock_env


def _drive(cb, success_value, n_rollouts):
    for _ in range(n_rollouts):
        cb.locals = {"infos": [{"reward/success": success_value}]}
        cb._on_step()
        cb._on_rollout_end()


class TestAdaptiveCurriculumCallback:
    def test_advances_one_level_on_high_success(self) -> None:
        cb, _ = _make_adaptive(start_idx=3)
        _drive(cb, 5.0, 3)
        assert cb._carry_floor == pytest.approx(0.6)

    def test_retreats_one_level_on_low_success(self) -> None:
        cb, _ = _make_adaptive(start_idx=3)
        _drive(cb, 0.0, 3)
        assert cb._carry_floor == pytest.approx(0.4)

    def test_holds_in_band(self) -> None:
        cb, _ = _make_adaptive(start_idx=3)
        _drive(cb, 1.5, 3)
        assert cb._carry_floor == pytest.approx(0.5)

    def test_throttle_between_adjustments(self) -> None:
        cb, _ = _make_adaptive(start_idx=0)
        _drive(cb, 5.0, 3)
        assert cb._carry_floor == pytest.approx(0.3)
        _drive(cb, 5.0, 2)
        assert cb._carry_floor == pytest.approx(0.3)
        _drive(cb, 5.0, 1)
        assert cb._carry_floor == pytest.approx(0.4)

    def test_clamps_to_top_level(self) -> None:
        cb, _ = _make_adaptive(start_idx=len(_LEVELS) - 1)
        _drive(cb, 5.0, 30)
        assert cb._carry_floor == pytest.approx(0.8)

    def test_clamps_to_bottom_level(self) -> None:
        cb, _ = _make_adaptive(start_idx=0)
        _drive(cb, 0.0, 30)
        assert cb._carry_floor == pytest.approx(0.2)

    def test_only_carry_floor_is_sent_gripped_only(self) -> None:
        cb, mock_env = _make_adaptive(start_idx=0)
        cb._apply(0.5)
        sent = mock_env.env_method.call_args_list[-1]
        assert sent.args == ("set_curriculum_params", 0.010, 0.5), (
            "gripped-only: set_curriculum_params takes (clearance, carry_floor), no p_from_table"
        )

    def test_training_start_applies_bottom_level(self) -> None:
        cb, mock_env = _make_adaptive(start_idx=0)
        cb._on_training_start()
        mock_env.env_method.assert_called_once_with("set_curriculum_params", 0.010, 0.2)

    def test_curriculum_never_stops_the_run(self) -> None:
        for idx in (0, len(_LEVELS) - 1):
            cb, _ = _make_adaptive(start_idx=idx)
            for step in (1000, 50_000_001, 500_000_000):
                cb.num_timesteps = step
                cb.locals = {"infos": []}
                assert cb._on_step() is True
