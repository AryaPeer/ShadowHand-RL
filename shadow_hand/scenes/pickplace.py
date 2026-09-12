from dataclasses import dataclass, field

import mujoco

from shadow_hand.config import PickPlaceSceneConfig
from shadow_hand.scenes.common import (
    SLIDE_Z_RANGE,
    SensorMap,
    add_fingertip_sites_and_sensors,
    add_hand_slider,
    add_workspace,
    attach_hand,
    init_spec_options,
    resolve_hand_names,
)


@dataclass
class PickPlaceNameMap:
    hand_joint_ids: list[int]
    hand_qpos_start: int
    hand_qpos_end: int
    hand_qvel_start: int
    hand_qvel_end: int

    palm_body_id: int
    fingertip_site_ids: list[int]
    finger_geom_ids_per_finger: list[set[int]]
    object_body_id: int
    object_geom_id: int
    obj_qpos_start: int
    obj_qvel_start: int
    goal_body_id: int
    goal_mocap_id: int
    sensor_map: SensorMap = field(default_factory=SensorMap.empty)


def build_pickplace_scene(
    config: PickPlaceSceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, PickPlaceNameMap]:

    if config is None:
        config = PickPlaceSceneConfig()

    spec = mujoco.MjSpec()
    init_spec_options(spec, config)
    add_workspace(spec, config)
    mount_site = add_hand_slider(spec, config, slide_z_range=SLIDE_Z_RANGE, xy_forcerange=30.0)
    attach_hand(spec, mount_site)
    add_fingertip_sites_and_sensors(spec)

    half = config.object_half_extent
    obj_body = spec.worldbody.add_body(
        name="object",
        pos=[0.0, 0.0, config.table_height + half],
    )
    obj_body.add_freejoint(name="object_freejoint")
    obj_body.add_geom(
        name="object_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half, half, half],
        mass=config.object_mass,
        friction=list(config.object_friction),
        rgba=[0.2, 0.6, 0.9, 1.0],
        contype=1,
        conaffinity=1,
        condim=4,
    )

    gx, gy = config.goal_nominal_xy
    goal_body = spec.worldbody.add_body(name="goal", pos=[gx, gy, config.table_height + 0.001])
    goal_body.mocap = True
    goal_body.add_geom(
        name="goal_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[config.goal_marker_radius, 0.001, 0.0],
        rgba=[0.2, 0.9, 0.3, 0.5],
        contype=0,
        conaffinity=0,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_names(model)

    return model, data, name_map


def _resolve_names(model: mujoco.MjModel) -> PickPlaceNameMap:
    hand = resolve_hand_names(model, exclude_joint="object_freejoint")

    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_start = model.jnt_qposadr[obj_jnt_id]
    obj_qvel_start = model.jnt_dofadr[obj_jnt_id]

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")

    goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
    goal_mocap_id = int(model.body_mocapid[goal_body_id])
    if goal_mocap_id < 0:
        raise ValueError("goal body is not a mocap body")

    return PickPlaceNameMap(
        hand_joint_ids=hand.hand_joint_ids,
        hand_qpos_start=hand.hand_qpos_start,
        hand_qpos_end=hand.hand_qpos_end,
        hand_qvel_start=hand.hand_qvel_start,
        hand_qvel_end=hand.hand_qvel_end,
        palm_body_id=hand.palm_body_id,
        fingertip_site_ids=hand.fingertip_site_ids,
        finger_geom_ids_per_finger=hand.finger_geom_ids_per_finger,
        object_body_id=object_body_id,
        object_geom_id=object_geom_id,
        obj_qpos_start=obj_qpos_start,
        obj_qvel_start=obj_qvel_start,
        goal_body_id=goal_body_id,
        goal_mocap_id=goal_mocap_id,
        sensor_map=SensorMap(finger_touch_adr=hand.finger_touch_adr),
    )
