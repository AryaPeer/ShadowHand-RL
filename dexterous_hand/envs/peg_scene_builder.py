import math
from dataclasses import dataclass, field

import mujoco

from dexterous_hand.config import PegSceneConfig
from dexterous_hand.envs._scene_common import (
    SensorMap,
    add_fingertip_sites_and_sensors,
    add_hand_slider,
    add_workspace,
    attach_hand,
    init_spec_options,
    resolve_hand_names,
)

PEG_SLIDE_Z_RANGE: tuple[float, float] = (-0.10, 0.15)
HOLE_WALL_THICKNESS: float = 0.005
PEG_MESH_SIDES: int = 16
N_BORE_WALLS: int = 12


def _prism_uservert(n_sides: int, radius: float, half_length: float) -> list[float]:
    verts: list[float] = []
    for z in (half_length, -half_length):
        for k in range(n_sides):
            a = 2.0 * math.pi * k / n_sides
            verts += [radius * math.cos(a), radius * math.sin(a), z]

    return verts


WALL_SENSOR_NAMES = [f"hole_wall_{k:02d}" for k in range(N_BORE_WALLS)] + ["hole_bottom"]


@dataclass
class PegNameMap:
    hand_joint_ids: list[int]
    hand_qpos_start: int
    hand_qpos_end: int
    hand_qvel_start: int
    hand_qvel_end: int

    palm_body_id: int
    fingertip_site_ids: list[int]
    finger_geom_ids_per_finger: list[set[int]]

    peg_body_id: int
    peg_geom_id: int
    peg_qpos_start: int
    peg_qvel_start: int
    hole_body_id: int
    sensor_map: SensorMap = field(default_factory=SensorMap.empty)


def build_peg_scene(
    config: PegSceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, PegNameMap]:

    if config is None:
        config = PegSceneConfig()

    spec = mujoco.MjSpec()
    init_spec_options(spec, config)
    add_workspace(spec, config)
    mount_site = add_hand_slider(spec, config, slide_z_range=PEG_SLIDE_Z_RANGE, xy_forcerange=15.0)
    attach_hand(spec, mount_site, collide_with_walls=True)
    add_fingertip_sites_and_sensors(spec)

    for wall_name in WALL_SENSOR_NAMES:
        spec.add_sensor(
            name=f"sensor_force_{wall_name}",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"site_{wall_name}",
        )

    peg_kwargs = {
        "contype": 3,
        "conaffinity": 3,
        "condim": 4,
        "solref": [0.005, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    }
    wall_kwargs = {
        "contype": 2,
        "conaffinity": 2,
        "condim": 4,
        "solref": [0.005, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    }

    peg_mesh = spec.add_mesh()
    peg_mesh.name = "peg_mesh"
    peg_mesh.uservert = _prism_uservert(PEG_MESH_SIDES, config.peg_radius, config.peg_half_length)

    peg_z = config.table_height + config.peg_half_length + 0.001
    peg_body = spec.worldbody.add_body(
        name="peg",
        pos=[0.0, 0.0, peg_z],
    )
    peg_body.add_freejoint(name="peg_freejoint")
    peg_body.add_geom(
        name="peg_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="peg_mesh",
        mass=config.peg_mass,
        friction=list(config.peg_friction),
        rgba=[0.8, 0.2, 0.2, 1.0],
        **peg_kwargs,
    )

    hole_x = config.hole_offset[0]
    hole_y = config.hole_offset[1]
    hole_z = config.table_height + config.hole_top_above_table
    hole_body = spec.worldbody.add_body(
        name="hole",
        pos=[hole_x, hole_y, hole_z],
    )

    cr = config.peg_radius + config.clearance
    wt = HOLE_WALL_THICKNESS
    wh = config.hole_depth / 2

    wall_rgba = [0.4, 0.4, 0.5, 1.0]

    ring_radius = cr + wt / 2
    tang_hw = (cr + wt) * math.tan(math.pi / N_BORE_WALLS) + wt

    def _ring_geom(
        name: str, radius: float, size: list[float], pos_z: float, quat_z: float
    ) -> None:
        c, s = math.cos(quat_z / 2.0), math.sin(quat_z / 2.0)
        hole_body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=[radius * math.cos(quat_z), radius * math.sin(quat_z), pos_z],
            quat=[c, 0.0, 0.0, s],
            rgba=wall_rgba,
            **wall_kwargs,
        )

    for k in range(N_BORE_WALLS):
        theta = 2.0 * math.pi * k / N_BORE_WALLS
        _ring_geom(f"hole_wall_{k:02d}", ring_radius, [wt / 2, tang_hw, wh], -wh, theta)

    hole_body.add_geom(
        name="hole_bottom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cr + wt, cr + wt, wt / 2],
        pos=[0.0, 0.0, -config.hole_depth],
        rgba=wall_rgba,
        **wall_kwargs,
    )

    pedestal_top = -config.hole_depth - wt / 2
    pedestal_bottom = -config.hole_top_above_table
    if pedestal_top > pedestal_bottom:
        ph = (pedestal_top - pedestal_bottom) / 2
        hole_body.add_geom(
            name="hole_pedestal",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cr + wt, cr + wt, ph],
            pos=[0.0, 0.0, pedestal_bottom + ph],
            rgba=wall_rgba,
            **wall_kwargs,
        )

    if config.cradle_depth > 0.0:
        cri = config.peg_radius + config.cradle_clearance
        cradle_ring_r = cri + wt / 2
        cradle_tang_hw = (cri + wt) * math.tan(math.pi / N_BORE_WALLS) + wt
        cradle_body = spec.worldbody.add_body(
            name="cradle",
            pos=[config.pick_center[0], config.pick_center[1], config.table_height],
        )

        for k in range(N_BORE_WALLS):
            theta = 2.0 * math.pi * k / N_BORE_WALLS
            c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
            cradle_body.add_geom(
                name=f"cradle_wall_{k:02d}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wt / 2, cradle_tang_hw, config.cradle_depth / 2],
                pos=[
                    cradle_ring_r * math.cos(theta),
                    cradle_ring_r * math.sin(theta),
                    config.cradle_depth / 2,
                ],
                quat=[c, 0.0, 0.0, s],
                rgba=[0.5, 0.45, 0.35, 1.0],
                **wall_kwargs,
            )

    for wall_name in WALL_SENSOR_NAMES:
        spec.add_pair(
            geomname1="peg_geom",
            geomname2=wall_name,
            friction=[0.2, 0.2, 0.005, 0.0001, 0.0001],
            condim=4,
            solref=[0.005, 1.0],
        )

    pad = 0.002
    for k in range(N_BORE_WALLS):
        theta = 2.0 * math.pi * k / N_BORE_WALLS
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        hole_body.add_site(
            name=f"site_hole_wall_{k:02d}",
            pos=[ring_radius * math.cos(theta), ring_radius * math.sin(theta), -wh],
            size=[wt / 2 + pad, tang_hw + pad, wh + pad],
            quat=[c, 0.0, 0.0, s],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            group=4,
        )

    hole_body.add_site(
        name="site_hole_bottom",
        pos=[0.0, 0.0, -config.hole_depth],
        size=[cr + wt + pad, cr + wt + pad, wt / 2 + pad],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        group=4,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_peg_names(model)

    return model, data, name_map


def _resolve_peg_names(model: mujoco.MjModel) -> PegNameMap:
    hand = resolve_hand_names(model, exclude_joint="peg_freejoint")

    peg_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    peg_qpos_start = model.jnt_qposadr[peg_jnt_id]
    peg_qvel_start = model.jnt_dofadr[peg_jnt_id]

    peg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
    hole_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole")

    wall_force_adr = []
    for wall_name in WALL_SENSOR_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_force_{wall_name}")
        if sid >= 0:
            wall_force_adr.append(int(model.sensor_adr[sid]))

    sensor_map = SensorMap(
        finger_touch_adr=hand.finger_touch_adr,
        wall_force_adr=wall_force_adr,
    )

    return PegNameMap(
        hand_joint_ids=hand.hand_joint_ids,
        hand_qpos_start=hand.hand_qpos_start,
        hand_qpos_end=hand.hand_qpos_end,
        hand_qvel_start=hand.hand_qvel_start,
        hand_qvel_end=hand.hand_qvel_end,
        palm_body_id=hand.palm_body_id,
        fingertip_site_ids=hand.fingertip_site_ids,
        finger_geom_ids_per_finger=hand.finger_geom_ids_per_finger,
        peg_body_id=peg_body_id,
        peg_geom_id=peg_geom_id,
        peg_qpos_start=peg_qpos_start,
        peg_qvel_start=peg_qvel_start,
        hole_body_id=hole_body_id,
        sensor_map=sensor_map,
    )
