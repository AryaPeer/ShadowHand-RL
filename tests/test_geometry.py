import pytest

pytest.importorskip("mujoco")
pytest.importorskip("jax.numpy")

import jax.numpy as jnp
import mujoco
import numpy as np

from shadow_hand.config import (
    PegRewardConfig,
    PegSceneConfig,
    RewardConfig,
    SceneConfig,
)
from shadow_hand.scenes.common import (
    CUBE_GRIP_BIAS,
    CUBE_GRIP_SPAWN_XY,
    apply_flexion_bias,
)
from shadow_hand.scenes.grasp import (
    OBJECT_TYPES,
    build_grasp_scene,
    get_object_half_height,
)
from shadow_hand.scenes.peg import build_peg_scene
from shadow_hand.utils.mjx_helpers import get_insertion_depth_jax


def _peg_length(cfg: PegSceneConfig) -> float:
    return cfg.peg_half_length * 2.0


def _measure_depth(cfg: PegSceneConfig, model, data, nm) -> float:
    return float(
        get_insertion_depth_jax(
            jnp.array(data.xpos),
            jnp.array(data.xmat),
            nm.peg_body_id,
            nm.hole_body_id,
            cfg.peg_half_length,
            cfg.peg_radius,
            cfg.peg_radius + cfg.clearance,
            cfg.hole_depth,
        )
    )


def _set_peg_pose(model, data, pos, quat) -> None:
    peg_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    ]
    data.qpos[peg_qadr : peg_qadr + 3] = pos
    data.qpos[peg_qadr + 3 : peg_qadr + 7] = quat
    data.qvel[:] = 0.0


def test_success_depth_fits_in_tube():
    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, _, _ = build_peg_scene(cfg)

    entrance_z = float(model.body("hole").pos[2])
    bottom_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_bottom")
    bottom_body = model.geom_bodyid[bottom_gid]
    bottom_top_z = float(
        model.body_pos[bottom_body][2]
        + model.geom_pos[bottom_gid][2]
        + model.geom_size[bottom_gid][2]
    )
    max_in_tube = min(entrance_z - bottom_top_z, cfg.hole_top_above_table)

    required = rcfg.success_threshold * _peg_length(cfg)
    assert required < max_in_tube


def test_peg_is_a_mesh_cylinder_that_fits_the_round_bore():
    from shadow_hand.scenes.peg import PEG_MESH_SIDES

    cfg = PegSceneConfig()
    model, _, nm = build_peg_scene(cfg)
    peg_gid = nm.peg_geom_id
    assert int(model.geom_type[peg_gid]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_id = int(model.geom_dataid[peg_gid])
    assert int(model.mesh_vertnum[mesh_id]) == 2 * PEG_MESH_SIDES, "prism = two n-gon rings"
    circum = (cfg.peg_radius**2 + cfg.peg_half_length**2) ** 0.5
    assert float(model.geom_rbound[peg_gid]) >= circum - 1e-4, "mesh does not span the prism"
    assert cfg.peg_radius < cfg.peg_radius + cfg.clearance, "peg must fit the bore with clearance"


def test_insertion_depth_requires_lateral_containment():
    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = _peg_length(cfg)
    required = rcfg.success_threshold * peg_len

    upright = [1.0, 0.0, 0.0, 0.0]
    lying = [0.7071068, 0.7071068, 0.0, 0.0]
    spawn_z = cfg.table_height + cfg.peg_half_length + 0.001
    hx, hy = cfg.hole_offset

    outside_poses = [
        ("upright at the pick centre", [cfg.pick_center[0], cfg.pick_center[1], spawn_z], upright),
        ("upright in table corner", [0.20, 0.10, spawn_z], upright),
        ("lying flat 10cm out", [hx + 0.10, hy, cfg.table_height + cfg.peg_radius + 0.001], lying),
    ]

    for deg in (3.0, 8.0):
        theta = np.deg2rad(90.0 - deg)
        axis = np.array([np.sin(theta), 0.0, np.cos(theta)])
        center = np.array([hx, hy, cfg.table_height + cfg.peg_radius]) + axis * cfg.peg_half_length
        quat_about_y = [np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]
        outside_poses.append(
            (f"under-tube, {deg:.0f}deg tilt, end below bore", center.tolist(), quat_about_y)
        )

    for _label, pos, quat in outside_poses:
        _set_peg_pose(model, data, pos, quat)
        mujoco.mj_forward(model, data)
        depth = _measure_depth(cfg, model, data, nm)
        assert depth == 0.0

    in_tube_z = float(model.body("hole").pos[2]) - cfg.hole_depth + 0.0025 + cfg.peg_half_length
    _set_peg_pose(model, data, [hx, hy, in_tube_z], upright)
    mujoco.mj_forward(model, data)
    depth = _measure_depth(cfg, model, data, nm)
    assert depth > required


def test_under_tube_slot_is_blocked():
    cfg = PegSceneConfig()
    model, _, _ = build_peg_scene(cfg)

    ped_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_pedestal")
    assert ped_gid >= 0, "hole_pedestal geom missing — the under-tube slot is open"

    body = model.geom_bodyid[ped_gid]
    body_z = float(model.body_pos[body][2])
    ped_top = body_z + float(model.geom_pos[ped_gid][2]) + float(model.geom_size[ped_gid][2])
    ped_bottom = body_z + float(model.geom_pos[ped_gid][2]) - float(model.geom_size[ped_gid][2])

    plate_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_bottom")
    plate_bottom = (
        body_z + float(model.geom_pos[plate_gid][2]) - float(model.geom_size[plate_gid][2])
    )

    assert ped_top >= plate_bottom - 1e-9
    assert ped_bottom <= cfg.table_height + 1e-9
    wall_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_wall_px")
    wall_outer_x = float(model.geom_pos[wall_gid][0]) + float(model.geom_size[wall_gid][0])
    assert float(model.geom_size[ped_gid][0]) >= wall_outer_x - 1e-9
    assert float(model.geom_size[ped_gid][1]) >= wall_outer_x - 1e-9


def test_compiled_scene_contact_options():
    for build, cfg in (
        (build_grasp_scene, SceneConfig()),
        (build_peg_scene, PegSceneConfig()),
    ):
        model = build(cfg)[0]
        assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        assert model.opt.impratio == 1.0
        assert model.opt.iterations == cfg.solver_iterations
        assert model.opt.ls_iterations == cfg.ls_iterations
        assert model.opt.timestep == cfg.sim_timestep
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_wall_touch_sensors_alive():
    import math

    from shadow_hand.scenes.peg import N_BORE_WALLS

    cfg = PegSceneConfig()
    model, data, nm = build_peg_scene(cfg)
    cr = cfg.peg_radius + cfg.clearance
    entrance_z = float(model.body("hole").pos[2])
    hx, hy = cfg.hole_offset
    upright = [1.0, 0.0, 0.0, 0.0]

    press = cr - cfg.peg_radius + 0.0006
    in_bore_z = entrance_z - 0.02 + cfg.peg_half_length
    poses = {}
    for k in (0, N_BORE_WALLS // 4, N_BORE_WALLS // 2):
        theta = 2.0 * math.pi * k / N_BORE_WALLS
        poses[f"hole_wall_{k:02d}"] = [
            hx + press * math.cos(theta),
            hy + press * math.sin(theta),
            in_bore_z,
        ]

    poses["hole_bottom"] = [
        hx,
        hy,
        entrance_z - cfg.hole_depth + 0.0025 + cfg.peg_half_length - 0.0003,
    ]

    for wall, pos in poses.items():
        _set_peg_pose(model, data, pos, upright)
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_force_{wall}")
        assert sid >= 0
        val = float(data.sensordata[model.sensor_adr[sid]])
        assert val > 0.0


@pytest.mark.slow
def test_peg_drop_insertion_reaches_success_depth():
    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = _peg_length(cfg)

    entrance_z = float(model.body("hole").pos[2])
    hx, hy = cfg.hole_offset
    _set_peg_pose(model, data, [hx, hy, entrance_z + 0.02], [1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    for _ in range(400):
        mujoco.mj_step(model, data)

    frac = _measure_depth(cfg, model, data, nm) / peg_len
    assert frac >= rcfg.success_threshold + 0.03

    fracs = []
    for _ in range(rcfg.peg_hold_steps * 5):
        mujoco.mj_step(model, data)
        fracs.append(_measure_depth(cfg, model, data, nm) / peg_len)

    assert np.min(fracs) >= rcfg.success_threshold


@pytest.mark.slow
def test_peg_transport_release_insertion():
    import numpy as np

    from shadow_hand.scenes.common import GRIP_BIAS, build_grip_ctrl

    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = _peg_length(cfg)

    qpos = data.qpos.copy()
    apply_flexion_bias(qpos, model, bias_map=GRIP_BIAS)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    _set_peg_pose(model, data, data.site_xpos[sid].copy(), [1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    grip = build_grip_ctrl(model)
    data.ctrl[:] = grip
    for _ in range(5):
        mujoco.mj_step(model, data)

    hole_pos = data.xpos[nm.hole_body_id].copy()
    entrance_z = hole_pos[2]

    def act(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def slide_xy() -> np.ndarray:
        out = []
        for n in ("slide_x", "slide_y"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            out.append(float(data.qpos[model.jnt_qposadr[jid]]))

        return np.array(out)

    z_cmd = 0.0
    xy_cmd = np.zeros(2)
    open_frac = 0.0

    def do_steps(n: int, open_fingers: bool = False, servo: bool = False) -> None:
        nonlocal xy_cmd, open_frac

        for _ in range(n):
            if open_fingers:
                open_frac = min(open_frac + 0.15, 1.0)
                c = grip * (1.0 - open_frac)
            else:
                c = grip.copy()

            if servo:
                err = hole_pos[:2] - data.xpos[nm.peg_body_id][:2]
                desired = slide_xy() + 0.4 * err
                xy_cmd = xy_cmd + np.clip(desired - xy_cmd, -0.003, 0.003)

            c[act("slide_x_act")] = xy_cmd[0]
            c[act("slide_y_act")] = xy_cmd[1]
            lo, hi = model.actuator_ctrlrange[act("slide_z_act")]
            c[act("slide_z_act")] = float(np.clip(z_cmd, lo, hi))
            data.ctrl[:] = c
            mujoco.mj_step(model, data, nstep=cfg.frame_skip)

    z_cmd = 0.06
    do_steps(15)
    xy_cmd = slide_xy() + (hole_pos[:2] - data.xpos[nm.peg_body_id][:2])
    do_steps(40)
    do_steps(40, servo=True)
    tip_z = data.xpos[nm.peg_body_id][2] - peg_len / 2.0
    z_cmd += entrance_z + 0.01 - tip_z
    do_steps(15, servo=True)

    for _ in range(10):
        tip_z = data.xpos[nm.peg_body_id][2] - peg_len / 2.0
        z_cmd += float(np.clip((entrance_z - 0.020) - tip_z, -0.004, 0.004))
        do_steps(2, servo=True)

    do_steps(15, open_fingers=True)
    z_cmd += 0.06
    do_steps(25, open_fingers=True)

    do_steps(75, open_fingers=True)

    frac = _measure_depth(cfg, model, data, nm) / peg_len
    assert frac >= rcfg.success_threshold + 0.03
    fracs = []
    for _ in range(50):
        do_steps(1, open_fingers=True)
        fracs.append(_measure_depth(cfg, model, data, nm) / peg_len)

    assert np.min(fracs) >= rcfg.success_threshold


CUBE_GRIP_SEED = {
    "sx": 0.115,
    "sy": -0.017,
    "z0": -0.02,
    "j3": 1.0,
    "j12": 0.5,
    "thj5": 0.5,
    "th1": 0.7,
    "squeeze": 0.4,
}
CUBE_SPAWN_XY = (0.075, 0.0)


def _grasp_set_joint(model, qpos, name: str, val: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    qpos[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))


def _grasp_seta(model, ctrl, act_name: str, target: float) -> None:
    ai = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    if ai < 0:
        return
    lo, hi = model.actuator_ctrlrange[ai]
    ctrl[ai] = float(np.clip(target, lo, hi))


def _cube_grip_ctrl(model, p: dict, squeeze: float, z: float) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=np.float64)
    _grasp_seta(model, ctrl, "slide_x_act", p["sx"])
    _grasp_seta(model, ctrl, "slide_y_act", p["sy"])
    _grasp_seta(model, ctrl, "slide_z_act", z)
    for an in ("rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"):
        _grasp_seta(model, ctrl, an, p["j3"] + squeeze)
    for an in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
        _grasp_seta(model, ctrl, an, p["j12"] * 2 + squeeze)
    _grasp_seta(model, ctrl, "rh_A_THJ5", p["thj5"])
    _grasp_seta(model, ctrl, "rh_A_THJ4", 1.2)
    _grasp_seta(model, ctrl, "rh_A_THJ2", 0.3)
    _grasp_seta(model, ctrl, "rh_A_THJ1", p["th1"] + squeeze)
    return ctrl


def test_pre_grasped_spawn_starts_with_formed_grip():
    gt, gs = OBJECT_TYPES["large_cube"]
    scfg = SceneConfig(object_half_extent=gs[0])
    model, data, nm = build_grasp_scene(scfg)

    qpos = data.qpos.copy()
    apply_flexion_bias(qpos, model, bias_map=CUBE_GRIP_BIAS)

    gt, gs = OBJECT_TYPES["large_cube"]
    obj_z0 = scfg.table_height + get_object_half_height(gt, gs) + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [CUBE_GRIP_SPAWN_XY[0], CUBE_GRIP_SPAWN_XY[1], obj_z0]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hand_geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        hand_geoms |= gset

    def cube_contacts() -> int:
        return sum(
            1
            for ci in range(data.ncon)
            if (
                data.contact[ci].geom1 == nm.object_geom_id and data.contact[ci].geom2 in hand_geoms
            )
            or (
                data.contact[ci].geom2 == nm.object_geom_id and data.contact[ci].geom1 in hand_geoms
            )
        )

    n_at_spawn = cube_contacts()
    assert n_at_spawn >= 2

    assert CUBE_GRIP_BIAS["slide_x"] == CUBE_GRIP_SEED["sx"]
    assert CUBE_GRIP_BIAS["slide_y"] == CUBE_GRIP_SEED["sy"]
    assert CUBE_GRIP_BIAS["slide_z"] == CUBE_GRIP_SEED["z0"]
    assert CUBE_GRIP_BIAS["rh_FFJ3"] == CUBE_GRIP_SEED["j3"]
    assert CUBE_GRIP_BIAS["rh_FFJ2"] == CUBE_GRIP_SEED["j12"]
    assert CUBE_GRIP_BIAS["rh_THJ5"] == CUBE_GRIP_SEED["thj5"]
    assert CUBE_GRIP_BIAS["rh_THJ1"] == CUBE_GRIP_SEED["th1"]
    assert tuple(CUBE_GRIP_SPAWN_XY) == tuple(CUBE_SPAWN_XY)


@pytest.mark.slow
def test_grasp_lift_reaches_target_height():
    scfg = SceneConfig()
    rcfg = RewardConfig()
    model, data, nm = build_grasp_scene(scfg)
    p = CUBE_GRIP_SEED

    qpos = data.qpos.copy()
    _grasp_set_joint(model, qpos, "slide_x", p["sx"])
    _grasp_set_joint(model, qpos, "slide_y", p["sy"])
    _grasp_set_joint(model, qpos, "slide_z", p["z0"])

    for j in ("FF", "MF", "RF", "LF"):
        _grasp_set_joint(model, qpos, f"rh_{j}J3", p["j3"])
        _grasp_set_joint(model, qpos, f"rh_{j}J2", p["j12"])
        _grasp_set_joint(model, qpos, f"rh_{j}J1", p["j12"])

    _grasp_set_joint(model, qpos, "rh_THJ5", p["thj5"])
    _grasp_set_joint(model, qpos, "rh_THJ4", 1.2)
    _grasp_set_joint(model, qpos, "rh_THJ2", 0.3)
    _grasp_set_joint(model, qpos, "rh_THJ1", p["th1"])

    gt, gs = OBJECT_TYPES["large_cube"]
    obj_z0 = scfg.table_height + get_object_half_height(gt, gs) + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [CUBE_SPAWN_XY[0], CUBE_SPAWN_XY[1], obj_z0]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hand_geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        hand_geoms |= gset

    lift_z = 0.18
    n_settle, n_lift, n_hold = 30, 40, 80
    total = n_settle + n_lift + n_hold
    final_lift = 0.0
    held = 0
    for step in range(total):
        if step < n_settle:
            squeeze, z = p["squeeze"] * min(step / 10.0, 1.0), p["z0"]
        elif step < n_settle + n_lift:
            t = (step - n_settle) / n_lift
            squeeze, z = p["squeeze"], p["z0"] + (lift_z - p["z0"]) * t
        else:
            squeeze, z = p["squeeze"], lift_z

        data.ctrl[: model.nu] = _cube_grip_ctrl(model, p, squeeze, z)
        mujoco.mj_step(model, data, nstep=scfg.frame_skip)
        final_lift = float(data.xpos[nm.object_body_id][2] - obj_z0)
        if step >= total - 40:
            ncon = sum(
                1
                for ci in range(data.ncon)
                if (
                    data.contact[ci].geom1 == nm.object_geom_id
                    and data.contact[ci].geom2 in hand_geoms
                )
                or (
                    data.contact[ci].geom2 == nm.object_geom_id
                    and data.contact[ci].geom1 in hand_geoms
                )
            )
            held += int(ncon > 0)

    assert final_lift >= rcfg.lift_target + 0.05
    assert held >= 39


def test_contact_mask_helper_excludes_non_object_geoms():
    from shadow_hand.utils.mjx_helpers import (
        get_finger_object_contact_mask,
        pad_id_groups,
    )

    finger_ids = pad_id_groups([{10, 11}, {20}, {30}])
    object_ids = jnp.asarray([99], dtype=jnp.int32)
    contact_geom = jnp.asarray([[10, 5], [99, 20], [30, 99]], dtype=jnp.int32)
    contact_dist = jnp.asarray([-0.001, -0.001, 0.002], dtype=jnp.float32)

    mask = get_finger_object_contact_mask(contact_geom, contact_dist, finger_ids, object_ids)
    assert mask.tolist() == [False, True, False]


def test_table_press_with_distant_cube_counts_zero_grasp_contacts():
    from shadow_hand.utils.mjx_helpers import (
        get_finger_object_contact_mask,
        pad_id_groups,
    )

    scfg = SceneConfig()
    model, data, nm = build_grasp_scene(scfg)

    gt, gs = OBJECT_TYPES["large_cube"]
    obj_z0 = scfg.table_height + get_object_half_height(gt, gs) + 0.001
    s = nm.obj_qpos_start
    data.qpos[s : s + 3] = [0.30, 0.28, obj_z0]

    zid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_z")
    zadr = model.jnt_qposadr[zid]
    data.qpos[zadr] = model.jnt_range[zid][0]
    apply_flexion_bias(data.qpos, model, bias_map=CUBE_GRIP_BIAS)
    data.qvel[:] = 0.0

    ctrl_low, ctrl_high = model.actuator_ctrlrange.T
    data.ctrl[:] = np.clip(0.0, ctrl_low, ctrl_high)
    for _ in range(200):
        mujoco.mj_step(model, data)

    hand_geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        hand_geoms |= gset
    table_contacts = sum(
        1
        for ci in range(data.ncon)
        if (data.contact[ci].geom1 in hand_geoms or data.contact[ci].geom2 in hand_geoms)
        and nm.object_geom_id not in (data.contact[ci].geom1, data.contact[ci].geom2)
    )
    assert table_contacts > 0, "setup failed: hand is not touching the table at all"

    from shadow_hand.utils.mjx_helpers import get_finger_touch_from_sensors

    _, sensor_mask = get_finger_touch_from_sensors(
        jnp.asarray(data.sensordata),
        jnp.asarray(nm.sensor_map.finger_touch_adr, dtype=jnp.int32),
    )
    assert int(sensor_mask.sum()) > 0, (
        "the touch sensors must fire in this state, or the guard below proves nothing"
    )

    mask = get_finger_object_contact_mask(
        jnp.asarray(data.contact.geom[: data.ncon], dtype=jnp.int32),
        jnp.asarray(data.contact.dist[: data.ncon], dtype=jnp.float32),
        pad_id_groups(nm.finger_geom_ids_per_finger),
        jnp.asarray([nm.object_geom_id], dtype=jnp.int32),
    )
    assert int(mask.sum()) == 0
