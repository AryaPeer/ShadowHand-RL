import jax.numpy as jnp

from shadow_hand.config import (
    PegRewardConfig,
    PegSceneConfig,
    PickPlaceRewardConfig,
    PickPlaceSceneConfig,
    RewardConfig,
)
from shadow_hand.rewards.grasp_reward import grasp_reward, init_grasp_reward_state
from shadow_hand.rewards.peg_reward import (
    PegRewardState,
    init_peg_reward_state,
    peg_reward,
)
from shadow_hand.rewards.pickplace_reward import init_pickplace_reward_state, pickplace_reward


def check_peg() -> bool:
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

    print("\n=== PEG reward gradient ===\n")

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

    print("  winning-trajectory per-step totals:")
    print(f"    reach-only (no grip)           = {t_reach:>9.3f}")
    print(f"    gripped on table (r={spawn_r * 100:.0f}cm)     = {t_table:>9.3f}")
    print(f"    gripped lifted 5cm             = {t_lift:>9.3f}")
    print(f"    gripped at engaged pose        = {t_carry:>9.3f}")
    print(f"    RELEASED at engaged pose       = {t_release_engaged:>9.3f}")
    print(f"    RELEASED, settled in bore      = {t_settled:>9.3f}")
    print(f"    HELD gripped at engaged (farm) = {t_hold:>9.3f}")
    print(f"    HELD gripped deep (frac 0.69)  = {t_farm:>9.3f}")
    print(f"    ungripped on table @ bore xy   = {t_false_bottom:>9.3f}")
    print(f"    parked by tube, NO grip        = {t_parked:>9.3f}")
    print(f"    DROPPED mid-carry (r=8cm)      = {t_drop_far:>9.3f}  (prem {prem_drop_far:+.2f})")
    print()

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

    print(f"  GATE 1: monotone table<lift<carry<settled       {'PASS' if monotone else 'FAIL'}")
    print(f"  GATE 2: release beats carry (no dip)            {'PASS' if anti_cliff else 'FAIL'}")
    print(
        f"  GATE 3: parked-ungripped pays ~nothing          "
        f"{'PASS' if parked_pays_nothing else 'FAIL'}"
    )
    print(
        f"  GATE 4: on-table @ bore pays ~nothing           {'PASS' if no_false_bottom else 'FAIL'}"
    )
    print(f"  GATE 5: engaged-release beats hover-and-hold    {'PASS' if anti_hold else 'FAIL'}")
    print(f"  GATE 6: released-settled is the global optimum  {'PASS' if dominance else 'FAIL'}")
    print(
        f"  GATE 7: mid-carry drop punished < holding       "
        f"{'PASS' if no_premature_drop else 'FAIL'}"
    )

    return (
        monotone
        and anti_cliff
        and parked_pays_nothing
        and no_false_bottom
        and anti_hold
        and dominance
        and no_premature_drop
    )


def check_grasp() -> bool:
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

    print("\n=== GRASP reward gradient ===\n")
    info_sit = run(initial_z + 0.0)
    info_lift = run(initial_z + cfg.lift_target)

    total_sit = float(info_sit["reward/total"])
    total_lift = float(info_lift["reward/total"])
    delta_total = total_lift - total_sit
    grasp_post_weight = float(info_sit["reward/grasping"])

    print("  at lift_height = 0mm (perfect grip, no lift):")
    print(f"    reward/total              = {total_sit:>8.4f}")
    print(f"    reward/grasping           = {grasp_post_weight:>8.4f}  (post-weight)")
    print(f"    reward/lifting            = {float(info_sit['reward/lifting']):>8.4f}")
    print()
    print(f"  at lift_height = {cfg.lift_target * 1000:.0f}mm (= lift_target):")
    print(f"    reward/total              = {total_lift:>8.4f}")
    print(f"    reward/lifting            = {float(info_lift['reward/lifting']):>8.4f}")
    print()
    print(f"  delta_total (lifting to target) = {delta_total:>+8.4f}")
    print(
        f"  delta_lift_component             = "
        f"{float(info_lift['reward/lifting']) - float(info_sit['reward/lifting']):>+8.4f}"
    )
    print()

    intermediate = run(initial_z + cfg.lift_target * 0.5)
    monotonic = (
        float(info_sit["reward/total"])
        < float(intermediate["reward/total"])
        < float(info_lift["reward/total"])
    )

    bar_delta = 5.0
    pass_delta = delta_total >= bar_delta

    print(f"  GATE 1: delta_total >= {bar_delta}             {'PASS' if pass_delta else 'FAIL'}")
    print(f"  GATE 2: monotonic 0 -> half -> full target  {'PASS' if monotonic else 'FAIL'}")
    return pass_delta and monotonic


def check_pickplace() -> bool:
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

    print("\n=== PICKPLACE reward gradient ===\n")
    print("  winning-trajectory per-step totals:")
    print(f"    reach-only (no grip)           = {t_reach:>9.3f}")
    print(f"    gripped @ source               = {t_grip:>9.3f}")
    print(f"    gripped lifted @ source        = {t_lift:>9.3f}")
    print(f"    carried, lifted over goal      = {t_hover:>9.3f}")
    print(f"    HELD gripped on table @ goal   = {t_hold:>9.3f}")
    print(f"    released @ goal (no annuity)   = {t_settled_nos:>9.3f}")
    print(f"    RELEASED, settled @ goal       = {t_settled:>9.3f}")
    print(f"    parked, no grip                = {t_parked:>9.3f}")
    print(f"    bulldozed @ goal (never lifted) placed = {placed_bulldoze:>7.4f}")
    print()

    monotone = t_reach < t_grip < t_lift < t_hover < t_settled
    anti_cliff = t_settled > t_hover and t_settled_nos > t_hover
    parked_pays_nothing = t_parked < 1.0 and t_parked < t_grip
    no_bulldoze = placed_bulldoze < 0.05
    anti_hold = t_settled_nos > t_hold
    dominance = t_settled >= max(t_reach, t_grip, t_lift, t_hover, t_hold, t_parked)

    print(f"  GATE 1: monotone reach<grip<lift<carry<settled   {'PASS' if monotone else 'FAIL'}")
    print(f"  GATE 2: settled beats carry (release, no dip)    {'PASS' if anti_cliff else 'FAIL'}")
    print(
        f"  GATE 3: parked-ungripped pays ~nothing           "
        f"{'PASS' if parked_pays_nothing else 'FAIL'}"
    )
    print(f"  GATE 4: bulldozing (never lifted) pays nothing   {'PASS' if no_bulldoze else 'FAIL'}")
    print(f"  GATE 5: release beats place-and-hold (no farm)   {'PASS' if anti_hold else 'FAIL'}")
    print(f"  GATE 6: released-settled is the global optimum   {'PASS' if dominance else 'FAIL'}")

    return (
        monotone and anti_cliff and parked_pays_nothing and no_bulldoze and anti_hold and dominance
    )


if __name__ == "__main__":
    peg_ok = check_peg()
    grasp_ok = check_grasp()
    pickplace_ok = check_pickplace()
    print()
    print("=" * 60)
    print(f"PEG:       {'PASS' if peg_ok else 'FAIL'}")
    print(f"GRASP:     {'PASS' if grasp_ok else 'FAIL'}")
    print(f"PICKPLACE: {'PASS' if pickplace_ok else 'FAIL'}")
    print("=" * 60)

    if not (peg_ok and grasp_ok and pickplace_ok):
        print("\nDO NOT spend on a full run — fix reward shape first.")
        raise SystemExit(1)

    print("\nReward gradient is correctly oriented. Sanity run is safe to launch.")
