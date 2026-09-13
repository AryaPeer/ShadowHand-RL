import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from shadow_hand.config import PickPlaceRewardConfig, PickPlaceSceneConfig
from shadow_hand.rewards.pickplace import (
    init_pickplace_reward_state,
    pickplace_reward,
)

_CFG = PickPlaceRewardConfig()
_SCFG = PickPlaceSceneConfig()
_TABLE_H = _SCFG.table_height
_HALF = _SCFG.object_half_extent
_REST_Z = _TABLE_H + _HALF
_GOAL = _SCFG.goal_nominal_xy


def _eval(obj_xy, obj_z, gripped, was_lifted=False, speed=0.0, steady_steps=1):
    state = init_pickplace_reward_state(_REST_Z + 0.001, _TABLE_H)
    state = state._replace(was_lifted=jnp.array(was_lifted))
    px, py = obj_xy
    if gripped:
        fp = jnp.array(
            [
                [px + 0.005, py, obj_z],
                [px - 0.005, py, obj_z],
                [px - 0.005, py + 0.005, obj_z],
                [px - 0.005, py - 0.005, obj_z],
                [px - 0.005, py - 0.01, obj_z],
            ]
        )
        mask = jnp.array([True, True, True, True, True])
    else:
        fp = jnp.tile(jnp.array([px, py, obj_z + 0.08]), (5, 1))
        mask = jnp.array([False, False, False, False, False])

    info: dict = {}
    for _ in range(steady_steps):
        _, state, info = pickplace_reward(
            state=state,
            finger_positions=fp,
            palm_position=jnp.array([px, py, obj_z]),
            object_position=jnp.array([px, py, obj_z]),
            object_linear_velocity=jnp.array([speed, 0.0, 0.0]),
            finger_contact_mask=mask,
            goal_xy=jnp.asarray(_GOAL),
            actions=jnp.zeros(23),
            table_height=_TABLE_H,
            object_half_extent=_HALF,
            cfg=_CFG,
        )

    return info


def test_release_at_goal_beats_hover():
    hover = _eval(_GOAL, _REST_Z + _CFG.lift_target, gripped=True)
    settled = _eval(
        _GOAL, _REST_Z, gripped=False, was_lifted=True, steady_steps=_CFG.place_hold_steps + 5
    )
    assert float(settled["reward/total"]) > float(hover["reward/total"])


def test_release_no_dip_before_annuity():
    hover = _eval(_GOAL, _REST_Z + _CFG.lift_target, gripped=True)
    settled_nos = _eval(_GOAL, _REST_Z, gripped=False, was_lifted=True, steady_steps=1)
    assert float(settled_nos["reward/total"]) > float(hover["reward/total"])


def test_release_beats_place_and_hold():
    held = _eval(_GOAL, _REST_Z, gripped=True, was_lifted=True)
    released = _eval(_GOAL, _REST_Z, gripped=False, was_lifted=True)
    assert float(released["reward/total"]) > float(held["reward/total"])


def test_bulldoze_pays_no_placed():
    info = _eval(_GOAL, _REST_Z, gripped=True, was_lifted=False)
    assert float(info["reward/placed"]) < 0.05


def test_success_annuity_requires_release():
    steps = _CFG.place_hold_steps + 5
    released = _eval(_GOAL, _REST_Z, gripped=False, was_lifted=True, steady_steps=steps)
    gripped = _eval(_GOAL, _REST_Z, gripped=True, was_lifted=True, steady_steps=steps)
    assert float(released["is_success"]) == 1.0
    assert float(gripped["is_success"]) == 0.0


def test_drop_penalty_only_away_from_goal():
    far_xy = (_GOAL[0] + 0.15, _GOAL[1])
    info_far = _eval(far_xy, _REST_Z, gripped=False, was_lifted=True)
    assert float(info_far["reward/drop"]) < 0.0
    info_goal = _eval(_GOAL, _REST_Z, gripped=False, was_lifted=True)
    assert float(info_goal["reward/drop"]) == 0.0
