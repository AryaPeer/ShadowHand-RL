import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from shadow_hand.config import PegSceneConfig, PickPlaceSceneConfig, SceneConfig
from shadow_hand.paths import ASSETS_DIR
from shadow_hand.utils.mujoco_helpers import get_joint_qpos_qvel_range

SLIDE_Z_RANGE: tuple[float, float] = (-0.05, 0.20)
SLIDE_Z_INIT: float = (SLIDE_Z_RANGE[0] + SLIDE_Z_RANGE[1]) / 2.0

FINGERTIP_BODIES = [
    "rh_ffdistal",
    "rh_mfdistal",
    "rh_rfdistal",
    "rh_lfdistal",
    "rh_thdistal",
]

FINGERTIP_SITE_NAMES = ["fftip", "mftip", "rftip", "lftip", "thtip"]

FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "rh_ffdistal": [0.0, 0.0, 0.026],
    "rh_mfdistal": [0.0, 0.0, 0.026],
    "rh_rfdistal": [0.0, 0.0, 0.026],
    "rh_lfdistal": [0.0, 0.0, 0.026],
    "rh_thdistal": [0.0, 0.0, 0.032],
}

FINGER_BODY_PREFIXES = ["rh_ff", "rh_mf", "rh_rf", "rh_lf", "rh_th"]
FINGER_TOUCH_SITE_NAMES = ["ff_touch", "mf_touch", "rf_touch", "lf_touch", "th_touch"]

TABLE_TASK_FLEXION_BIAS: dict[str, float] = {
    "rh_FFJ3": 1.2,
    "rh_MFJ3": 1.2,
    "rh_RFJ3": 1.2,
    "rh_LFJ3": 1.2,
    "rh_FFJ2": 1.0,
    "rh_MFJ2": 1.0,
    "rh_RFJ2": 1.0,
    "rh_LFJ2": 1.0,
    "rh_THJ4": 1.2,
    "rh_THJ1": 1.0,
}


GRIP_BIAS: dict[str, float] = {
    "rh_FFJ3": 1.5,
    "rh_MFJ3": 1.5,
    "rh_RFJ3": 1.5,
    "rh_LFJ3": 1.5,
    "rh_FFJ2": 1.4,
    "rh_MFJ2": 1.4,
    "rh_RFJ2": 1.4,
    "rh_LFJ2": 1.4,
    "rh_FFJ1": 1.4,
    "rh_MFJ1": 1.4,
    "rh_RFJ1": 1.4,
    "rh_LFJ1": 1.4,
    "rh_THJ5": 1.047,
    "rh_THJ4": 1.2,
    "rh_THJ2": 0.5,
    "rh_THJ1": 1.4,
}

CUBE_GRIP_SPAWN_XY: tuple[float, float] = (0.075, 0.0)

CUBE_GRIP_BIAS: dict[str, float] = {
    "slide_x": 0.115,
    "slide_y": -0.017,
    "slide_z": -0.02,
    "rh_FFJ3": 1.0,
    "rh_MFJ3": 1.0,
    "rh_RFJ3": 1.0,
    "rh_LFJ3": 1.0,
    "rh_FFJ2": 0.5,
    "rh_MFJ2": 0.5,
    "rh_RFJ2": 0.5,
    "rh_LFJ2": 0.5,
    "rh_FFJ1": 0.5,
    "rh_MFJ1": 0.5,
    "rh_RFJ1": 0.5,
    "rh_LFJ1": 0.5,
    "rh_THJ5": 0.5,
    "rh_THJ4": 1.2,
    "rh_THJ2": 0.3,
    "rh_THJ1": 0.7,
}


def apply_flexion_bias(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    bias_map: dict[str, float] = TABLE_TASK_FLEXION_BIAS,
) -> np.ndarray:
    for jname, bias in bias_map.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        adr = int(model.jnt_qposadr[jid])
        low, high = model.jnt_range[jid]
        qpos[adr] = float(np.clip(bias, low, high))

    return qpos


def build_grip_ctrl(
    model: mujoco.MjModel,
    bias_map: dict[str, float] = GRIP_BIAS,
) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for ai in range(model.nu):
        trntype = int(model.actuator_trntype[ai])
        lo, hi = model.actuator_ctrlrange[ai]

        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[ai, 0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)

            if jname in bias_map:
                ctrl[ai] = float(np.clip(bias_map[jname], lo, hi))

        elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
            tid = int(model.actuator_trnid[ai, 0])
            wrap_start = int(model.tendon_adr[tid])
            wrap_count = int(model.tendon_num[tid])
            target = 0.0

            for wi in range(wrap_start, wrap_start + wrap_count):
                if int(model.wrap_type[wi]) != mujoco.mjtWrap.mjWRAP_JOINT:
                    continue
                jid = int(model.wrap_objid[wi])
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if jname in bias_map:
                    target += bias_map[jname]

            if target > 0.0:
                ctrl[ai] = float(np.clip(target, lo, hi))

    return ctrl


@dataclass
class SensorMap:
    finger_touch_adr: list[int]
    wall_force_adr: list[int] = field(default_factory=list)

    @staticmethod
    def empty() -> "SensorMap":
        return SensorMap(finger_touch_adr=[], wall_force_adr=[])


def init_spec_options(
    spec: mujoco.MjSpec, config: SceneConfig | PegSceneConfig | PickPlaceSceneConfig
) -> None:
    spec.option.timestep = config.sim_timestep
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    spec.option.iterations = config.solver_iterations
    spec.option.ls_iterations = config.ls_iterations
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if config.mjx_max_geom_pairs is not None:
        spec.add_numeric(name="max_geom_pairs", data=[float(config.mjx_max_geom_pairs)])
    if config.mjx_max_contact_points is not None:
        spec.add_numeric(name="max_contact_points", data=[float(config.mjx_max_contact_points)])
    spec.stat.extent = 1.0
    spec.stat.center = [0.0, 0.0, config.table_height]


def add_workspace(
    spec: mujoco.MjSpec, config: SceneConfig | PegSceneConfig | PickPlaceSceneConfig
) -> None:
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.01],
        rgba=[0.3, 0.3, 0.3, 1.0],
        conaffinity=1,
        condim=3,
    )

    table_half_h = 0.02
    table_body = spec.worldbody.add_body(
        name="table",
        pos=[0.0, 0.0, config.table_height - table_half_h],
    )
    table_body.add_geom(
        name="table_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[config.table_half_size, config.table_half_size, table_half_h],
        rgba=[0.5, 0.35, 0.2, 1.0],
        conaffinity=1,
        condim=3,
    )

    spec.worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        specular=[0.3, 0.3, 0.3],
    )
    spec.worldbody.add_camera(
        name="track_cam",
        pos=[0.8, -0.8, 0.8],
        xyaxes=[0.707, 0.707, 0.0, -0.354, 0.354, 0.866],
    )


def add_hand_slider(
    spec: mujoco.MjSpec,
    config: SceneConfig | PegSceneConfig | PickPlaceSceneConfig,
    *,
    slide_z_range: tuple[float, float],
    xy_forcerange: float,
) -> mujoco.MjsSite:
    slider = spec.worldbody.add_body(
        name="hand_slider",
        pos=[config.mount_x, config.mount_y, config.mount_height],
    )
    slider.add_joint(
        name="slide_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[-0.15, 0.15],
    )

    slider.add_joint(
        name="slide_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        range=[-0.15, 0.15],
    )

    slider.add_joint(
        name="slide_z",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[slide_z_range[0], slide_z_range[1]],
    )

    mount = slider.add_body(
        name="hand_mount",
        euler=[math.pi, 0.0, 0.0],
    )
    mount_site = mount.add_site(name="hand_attach", pos=[0.0, 0.0, 0.0])

    for name, target in (("slide_x_act", "slide_x"), ("slide_y_act", "slide_y")):
        spec.add_actuator(
            name=name,
            target=target,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            gainprm=[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            biasprm=[0, -100, -10, 0, 0, 0, 0, 0, 0, 0],
            ctrlrange=[-0.15, 0.15],
            forcerange=[-xy_forcerange, xy_forcerange],
        )

    spec.add_actuator(
        name="slide_z_act",
        target="slide_z",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[8000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -8000, -250, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[slide_z_range[0], slide_z_range[1]],
        forcerange=[-250, 250],
    )
    return mount_site


def attach_hand(
    spec: mujoco.MjSpec, mount_site: mujoco.MjsSite, *, collide_with_walls: bool = False
) -> None:
    hand_xml = str(ASSETS_DIR / "right_hand.xml")
    child_spec = mujoco.MjSpec.from_file(hand_xml)
    spec.attach(child_spec, site=mount_site, prefix="")
    spec.body("rh_forearm").quat = [0.0, 1.0, 0.0, 0.0]

    if collide_with_walls:
        for geom in spec.geoms:
            if geom.contype == 1 and geom.conaffinity == 0:
                geom.conaffinity = 2

    for mesh in spec.meshes:
        mesh.maxhullvert = 32


def add_fingertip_sites_and_sensors(spec: mujoco.MjSpec) -> None:
    for body_name, site_name in zip(FINGERTIP_BODIES, FINGERTIP_SITE_NAMES, strict=True):
        body = spec.body(body_name)
        offset = FINGERTIP_OFFSETS[body_name]
        body.add_site(
            name=site_name,
            pos=offset,
            size=[0.005],
            rgba=[1.0, 0.0, 0.0, 1.0],
        )

    for body_name, touch_site in zip(FINGERTIP_BODIES, FINGER_TOUCH_SITE_NAMES, strict=True):
        body = spec.body(body_name)
        offset = FINGERTIP_OFFSETS[body_name]
        body.add_site(
            name=touch_site,
            pos=offset,
            size=[0.012],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            group=4,
        )

    for touch_site in FINGER_TOUCH_SITE_NAMES:
        spec.add_sensor(
            name=f"sensor_{touch_site}",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=touch_site,
        )


@dataclass
class HandNames:
    hand_joint_ids: list[int]
    hand_qpos_start: int
    hand_qpos_end: int
    hand_qvel_start: int
    hand_qvel_end: int
    palm_body_id: int
    fingertip_site_ids: list[int]
    finger_geom_ids_per_finger: list[set[int]]
    finger_touch_adr: list[int]


def resolve_hand_names(model: mujoco.MjModel, *, exclude_joint: str) -> HandNames:
    hand_joint_ids = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != exclude_joint:
            hand_joint_ids.append(jid)

    if hand_joint_ids:
        hand_qpos_start, hand_qpos_end, hand_qvel_start, hand_qvel_end = get_joint_qpos_qvel_range(
            model, hand_joint_ids
        )
    else:
        hand_qpos_start = hand_qpos_end = 0
        hand_qvel_start = hand_qvel_end = 0

    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")

    fingertip_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FINGERTIP_SITE_NAMES
    ]

    finger_geom_ids_per_finger: list[set[int]] = [set() for _ in FINGER_BODY_PREFIXES]
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        for finger_idx, prefix in enumerate(FINGER_BODY_PREFIXES):
            if body_name.startswith(prefix):
                finger_geom_ids_per_finger[finger_idx].add(gid)
                break

    finger_touch_adr = []
    for touch_site in FINGER_TOUCH_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_{touch_site}")
        if sid >= 0:
            finger_touch_adr.append(int(model.sensor_adr[sid]))

    return HandNames(
        hand_joint_ids=hand_joint_ids,
        hand_qpos_start=hand_qpos_start,
        hand_qpos_end=hand_qpos_end,
        hand_qvel_start=hand_qvel_start,
        hand_qvel_end=hand_qvel_end,
        palm_body_id=palm_body_id,
        fingertip_site_ids=fingertip_site_ids,
        finger_geom_ids_per_finger=finger_geom_ids_per_finger,
        finger_touch_adr=finger_touch_adr,
    )
