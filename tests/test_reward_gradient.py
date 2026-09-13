import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from shadow_hand.config import (
    PegRewardConfig,
    PegSceneConfig,
    PickPlaceRewardConfig,
    PickPlaceSceneConfig,
    RewardConfig,
)
from shadow_hand.rewards.grasp import grasp_reward, init_grasp_reward_state
from shadow_hand.rewards.peg import (
    PegRewardState,
    init_peg_reward_state,
    peg_reward,
)
from shadow_hand.rewards.pickplace import init_pickplace_reward_state, pickplace_reward


def test_peg_reward_gradient() -> None:
    cfg = PegRewardConfig()
    scene = PegSceneConfig()
    pl = scene.peg_half_length * 2.0
    table_h = scene.table_height
    hole_z = table_h + scene.hole_top_above_table
    initial_z = table_h + scene.peg_half_length + 0.001

    def _step(
        state: PegRewardState,
        peg_z: float,
        peg_xy: tuple[float, float],
        gripped: bool,
        insertion_depth: float,
        stage: int,
    ) -> tuple:
        px, py = peg_xy
        if gripped:
            fp = jnp.array(
                [
                    [px + 0.005, py, peg_z],
                    [px - 0.005, py, peg_z],
                    [px - 0.005, py + 0.005, peg_z],
                    [px - 0.005, py - 0.005, peg_z],
                    [px - 0.005, py - 0.01, peg_z],
                ]
            )
            mask = jnp.array([True, True, True, True, True])
        else:
            fp = jnp.tile(jnp.array([px, py, peg_z + 0.08]), (5, 1))
            mask = jnp.array([False] * 5)

        _, state, info = peg_reward(
            state=state,
            stage=jnp.asarray(stage),
            finger_positions=fp,
            peg_position=jnp.array([px, py, peg_z]),
            peg_axis=jnp.array([0.0, 0.0, 1.0]),
            peg_linear_velocity=jnp.zeros(3),
            hole_position=jnp.array([0.0, 0.0, hole_z]),
            hole_axis=jnp.array([0.0, 0.0, 1.0]),
            insertion_depth=jnp.asarray(insertion_depth),
            contact_force_magnitude=jnp.asarray(0.0),
            finger_contact_mask=mask,
            peg_height=jnp.asarray(peg_z),
            actions=jnp.zeros(23),
            peg_length=pl,
            table_height=table_h,
            cfg=cfg,
        )
        return state, info

    def run(
        peg_z: float,
        peg_xy: tuple[float, float] = (0.0, 0.0),
        gripped: bool = True,
        insertion_depth: float = 0.0,
        stage: int = 1,
        steady_steps: int = 1,
    ) -> dict:
        state = init_peg_reward_state(initial_z)
        info: dict = {}
        for _ in range(steady_steps):
            state, info = _step(state, peg_z, peg_xy, gripped, insertion_depth, stage)
        return info

    def run_drop_far(release_r: float) -> dict:
        state = init_peg_reward_state(initial_z)
        state, _ = _step(state, initial_z, (0.04, 0.0), True, 0.0, 1)
        _, info = _step(state, initial_z, (release_r, 0.0), False, 0.0, 1)
        return info

    spawn_r = scene.spawn_min_radius
    engaged_z = hole_z + cfg.release_height + pl / 2.0
    engaged_depth = max(0.0, -cfg.release_height)
    settled_depth = scene.hole_depth - 0.0025
    settled_z = hole_z - settled_depth + pl / 2.0
    farm_depth = 0.69 * pl
    farm_z = hole_z - farm_depth + pl / 2.0

    s_reach = run(initial_z, peg_xy=(spawn_r, 0.0), gripped=False)
    s_table = run(initial_z, peg_xy=(spawn_r, 0.0))
    s_lift = run(initial_z + 0.05, peg_xy=(spawn_r, 0.0), stage=2)
    s_carry = run(engaged_z, stage=3, insertion_depth=engaged_depth)
    s_release_engaged = run(engaged_z, gripped=False, insertion_depth=engaged_depth, stage=3)
    s_settled = run(
        settled_z,
        gripped=False,
        insertion_depth=settled_depth,
        stage=3,
        steady_steps=cfg.peg_hold_steps + 5,
    )
    s_hold = run(engaged_z, stage=3, insertion_depth=engaged_depth, steady_steps=30)
    s_farm = run(farm_z, insertion_depth=farm_depth, stage=3, steady_steps=30)
    s_false_bottom = run(initial_z, peg_xy=(0.0, 0.0), gripped=False, stage=0, steady_steps=30)
    s_parked = run(initial_z, peg_xy=(0.05, 0.0), gripped=False, stage=0, steady_steps=30)
    s_drop_far = run_drop_far(0.08)

    t_reach = float(s_reach["reward/total"])
    t_table = float(s_table["reward/total"])
    t_lift = float(s_lift["reward/total"])
    t_carry = float(s_carry["reward/total"])
    t_release_engaged = float(s_release_engaged["reward/total"])
    t_settled = float(s_settled["reward/total"])
    t_hold = float(s_hold["reward/total"])
    t_farm = float(s_farm["reward/total"])
    t_false_bottom = float(s_false_bottom["reward/total"])
    t_parked = float(s_parked["reward/total"])
    t_drop_far = float(s_drop_far["reward/total"])
    prem_drop_far = float(s_drop_far["reward/premature_release"])

    monotone = t_table < t_lift < t_carry < t_settled
    anti_cliff = t_release_engaged >= t_carry and t_settled > t_carry
    parked_pays_nothing = t_parked < 1.0 and t_parked < t_table
    no_false_bottom = t_false_bottom < 1.0
    anti_hold = t_settled > t_hold and t_release_engaged > t_hold
    dominance = t_settled >= max(
        t_reach,
        t_table,
        t_lift,
        t_carry,
        t_release_engaged,
        t_hold,
        t_farm,
        t_false_bottom,
        t_parked,
    )
    no_premature_drop = prem_drop_far < 0.0 and t_drop_far < t_table

    assert monotone, "monotone table<lift<carry<settled"
    assert anti_cliff, "release beats carry (no dip)"
    assert parked_pays_nothing, "parked-ungripped pays ~nothing"
    assert no_false_bottom, "on-table @ bore pays ~nothing"
    assert anti_hold, "engaged-release beats hover-and-hold"
    assert dominance, "released-settled is the global optimum"
    assert no_premature_drop, "mid-carry drop punished < holding"


def test_grasp_reward_gradient() -> None:
    cfg = RewardConfig()
    table_h = 0.4
    initial_z = 0.43

    def run(obj_z: float) -> dict:
        state = init_grasp_reward_state(initial_z, table_h)
        fp = jnp.array(
            [
                [+0.005, 0.0, obj_z],
                [-0.005, 0.0, obj_z],
                [-0.005, 0.005, obj_z],
                [-0.005, -0.005, obj_z],
                [-0.005, -0.01, obj_z],
            ]
        )
        _, _, info = grasp_reward(
            state=state,
            finger_positions=fp,
            object_position=jnp.array([0.0, 0.0, obj_z]),
            object_linear_velocity=jnp.zeros(3),
            finger_contact_mask=jnp.array([True, True, True, True, True]),
            actions=jnp.zeros(23),
            table_height=table_h,
            cfg=cfg,
        )
        return info

    info_sit = run(initial_z + 0.0)
    info_lift = run(initial_z + cfg.lift_target)

    total_sit = float(info_sit["reward/total"])
    total_lift = float(info_lift["reward/total"])
    delta_total = total_lift - total_sit

    intermediate = run(initial_z + cfg.lift_target * 0.5)
    monotonic = (
        float(info_sit["reward/total"])
        < float(intermediate["reward/total"])
        < float(info_lift["reward/total"])
    )

    bar_delta = 5.0
    pass_delta = delta_total >= bar_delta

    assert pass_delta, f"delta_total >= {bar_delta}"
    assert monotonic, "monotonic 0 -> half -> full target"


def test_pickplace_reward_gradient() -> None:
    cfg = PickPlaceRewardConfig()
    scfg = PickPlaceSceneConfig()
    table_h = scfg.table_height
    half = scfg.object_half_extent
    rest_z = table_h + half
    initial_z = rest_z + 0.001
    source_xy = (0.075, 0.0)
    goal_xy = scfg.goal_nominal_xy

    def run(
        obj_xy: tuple[float, float],
        obj_z: float,
        gripped: bool,
        was_lifted: bool = False,
        steady_steps: int = 1,
    ) -> dict:
        state = init_pickplace_reward_state(initial_z, table_h)
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
            mask = jnp.array([False] * 5)

        info: dict = {}
        for _ in range(steady_steps):
            _, state, info = pickplace_reward(
                state=state,
                finger_positions=fp,
                palm_position=jnp.array([px, py, obj_z]),
                object_position=jnp.array([px, py, obj_z]),
                object_linear_velocity=jnp.zeros(3),
                finger_contact_mask=mask,
                goal_xy=jnp.asarray(goal_xy),
                actions=jnp.zeros(23),
                table_height=table_h,
                object_half_extent=half,
                cfg=cfg,
            )

        return info

    s_reach = run(source_xy, rest_z, gripped=False)
    s_grip = run(source_xy, rest_z, gripped=True)
    s_lift = run(source_xy, rest_z + cfg.lift_target, gripped=True)
    s_hover = run(goal_xy, rest_z + cfg.lift_target, gripped=True)
    s_settled = run(
        goal_xy, rest_z, gripped=False, was_lifted=True, steady_steps=cfg.place_hold_steps + 5
    )
    s_settled_nos = run(goal_xy, rest_z, gripped=False, was_lifted=True, steady_steps=1)
    s_bulldoze = run(goal_xy, rest_z, gripped=True, was_lifted=False)
    s_hold = run(goal_xy, rest_z, gripped=True, was_lifted=True)
    s_parked = run(source_xy, rest_z, gripped=False, steady_steps=30)

    t_reach = float(s_reach["reward/total"])
    t_grip = float(s_grip["reward/total"])
    t_lift = float(s_lift["reward/total"])
    t_hover = float(s_hover["reward/total"])
    t_settled = float(s_settled["reward/total"])
    t_settled_nos = float(s_settled_nos["reward/total"])
    t_hold = float(s_hold["reward/total"])
    t_parked = float(s_parked["reward/total"])
    placed_bulldoze = float(s_bulldoze["reward/placed"])

    monotone = t_reach < t_grip < t_lift < t_hover < t_settled
    anti_cliff = t_settled > t_hover and t_settled_nos > t_hover
    parked_pays_nothing = t_parked < 1.0 and t_parked < t_grip
    no_bulldoze = placed_bulldoze < 0.05
    anti_hold = t_settled_nos > t_hold
    dominance = t_settled >= max(t_reach, t_grip, t_lift, t_hover, t_hold, t_parked)

    assert monotone, "monotone reach<grip<lift<carry<settled"
    assert anti_cliff, "settled beats carry (release, no dip)"
    assert parked_pays_nothing, "parked-ungripped pays ~nothing"
    assert no_bulldoze, "bulldozing (never lifted) pays nothing"
    assert anti_hold, "release beats place-and-hold (no farm)"
    assert dominance, "released-settled is the global optimum"
