from dataclasses import dataclass, field

import mujoco

from shadow_hand.config import SceneConfig
from shadow_hand.envs._scene_common import (  # noqa: F401  (re-exported API)
    ASSETS_DIR,
    CUBE_GRIP_BIAS,
    CUBE_GRIP_SPAWN_XY,
    FINGER_BODY_PREFIXES,
    FINGER_TOUCH_SITE_NAMES,
    FINGERTIP_BODIES,
    FINGERTIP_OFFSETS,
    FINGERTIP_SITE_NAMES,
    GRIP_BIAS,
    TABLE_TASK_FLEXION_BIAS,
    SensorMap,
    add_fingertip_sites_and_sensors,
    add_hand_slider,
    add_workspace,
    apply_flexion_bias,
    attach_hand,
    build_grip_ctrl,
    init_spec_options,
    resolve_hand_names,
)

OBJECT_TYPES: dict[str, tuple[int, list[float]]] = {
    "large_cube": (mujoco.mjtGeom.mjGEOM_BOX, [0.035, 0.035, 0.035]),
}

SLIDE_Z_RANGE: tuple[float, float] = (-0.05, 0.20)
SLIDE_Z_INIT: float = (SLIDE_Z_RANGE[0] + SLIDE_Z_RANGE[1]) / 2.0


@dataclass
class NameMap:
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
    sensor_map: SensorMap = field(default_factory=SensorMap.empty)


def build_scene(
    config: SceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, NameMap]:

    if config is None:
        config = SceneConfig()

    spec = mujoco.MjSpec()
    init_spec_options(spec, config)
    add_workspace(spec, config)
    mount_site = add_hand_slider(spec, config, slide_z_range=SLIDE_Z_RANGE, xy_forcerange=30.0)
    attach_hand(spec, mount_site)
    add_fingertip_sites_and_sensors(spec)

    default_type, _ = OBJECT_TYPES["large_cube"]
    default_size = [config.object_half_extent] * 3
    obj_body = spec.worldbody.add_body(
        name="object",
        pos=[0.0, 0.0, config.table_height + default_size[2]],
    )
    obj_body.add_freejoint(name="object_freejoint")
    obj_body.add_geom(
        name="object_geom",
        type=default_type,
        size=default_size,
        mass=config.object_mass,
        friction=list(config.object_friction),
        rgba=[0.2, 0.6, 0.9, 1.0],
        contype=1,
        conaffinity=1,
        condim=4,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_names(model)

    return model, data, name_map


def _resolve_names(model: mujoco.MjModel) -> NameMap:
    hand = resolve_hand_names(model, exclude_joint="object_freejoint")

    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_start = model.jnt_qposadr[obj_jnt_id]
    obj_qvel_start = model.jnt_dofadr[obj_jnt_id]

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")

    return NameMap(
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
        sensor_map=SensorMap(finger_touch_adr=hand.finger_touch_adr),
    )


def get_object_half_height(geom_type: int, geom_size: list[float]) -> float:
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return geom_size[2]
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return geom_size[0]
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return geom_size[1]
    return 0.03
