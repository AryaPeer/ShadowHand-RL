from typing import Any

import jax.numpy as jnp


def get_contact_arrays(mjx_data: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    impl = getattr(mjx_data, "_impl", None)
    contact = impl.contact if impl is not None else mjx_data.contact
    return contact.geom, contact.dist


def get_finger_touch_from_sensors(
    sensordata: jnp.ndarray,
    finger_touch_adr: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    touch_vals = sensordata[finger_touch_adr]
    contact_mask = touch_vals > 0.0
    return touch_vals, contact_mask


def get_finger_object_contact_mask(
    contact_geom: jnp.ndarray,
    contact_dist: jnp.ndarray,
    finger_geom_ids: jnp.ndarray,
    object_geom_ids: jnp.ndarray,
) -> jnp.ndarray:
    g1 = contact_geom[:, 0]
    g2 = contact_geom[:, 1]
    active = contact_dist < 0.0

    is_obj1 = (g1[:, None] == object_geom_ids[None, :]).any(axis=-1)
    is_obj2 = (g2[:, None] == object_geom_ids[None, :]).any(axis=-1)
    is_fin1 = (g1[None, :, None] == finger_geom_ids[:, None, :]).any(axis=-1)
    is_fin2 = (g2[None, :, None] == finger_geom_ids[:, None, :]).any(axis=-1)

    pair = (is_fin1 & is_obj2[None, :]) | (is_fin2 & is_obj1[None, :])
    return (pair & active[None, :]).any(axis=-1)


def pad_id_groups(groups: list[set[int]]) -> jnp.ndarray:
    width = max((len(g) for g in groups), default=0)
    width = max(width, 1)
    rows = [sorted(g) + [-1] * (width - len(g)) for g in groups]
    return jnp.asarray(rows, dtype=jnp.int32)


def get_object_state_jax(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    xpos: jnp.ndarray,
    obj_body_id: int,
    obj_qpos_start: int,
    obj_qvel_start: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    position = xpos[obj_body_id]
    orientation = qpos[obj_qpos_start + 3 : obj_qpos_start + 7]
    linear_vel = qvel[obj_qvel_start : obj_qvel_start + 3]
    angular_vel = qvel[obj_qvel_start + 3 : obj_qvel_start + 6]
    return position, orientation, linear_vel, angular_vel


def get_fingertip_positions_jax(
    site_xpos: jnp.ndarray,
    fingertip_site_ids: jnp.ndarray,
) -> jnp.ndarray:

    return site_xpos[fingertip_site_ids]


def get_palm_position_jax(
    xpos: jnp.ndarray,
    palm_body_id: int,
) -> jnp.ndarray:

    return xpos[palm_body_id]


def get_body_axis_jax(
    xmat: jnp.ndarray,
    body_id: int,
    axis: int = 2,
) -> jnp.ndarray:

    return xmat[body_id].reshape(3, 3)[:, axis]


def get_insertion_depth_jax(
    xpos: jnp.ndarray,
    xmat: jnp.ndarray,
    peg_body_id: int,
    hole_body_id: int,
    peg_half_length: float,
    peg_radius: float,
    bore_radius: float,
    hole_depth: float,
) -> jnp.ndarray:

    peg_pos = xpos[peg_body_id]
    hole_pos = xpos[hole_body_id]
    hole_axis = xmat[hole_body_id].reshape(3, 3)[:, 2]
    peg_axis = xmat[peg_body_id].reshape(3, 3)[:, 2]

    axis_dot = jnp.dot(peg_axis, hole_axis)
    sin_tilt = jnp.sqrt(jnp.maximum(1.0 - axis_dot * axis_dot, 0.0))
    depth_of_center = jnp.dot(hole_pos - peg_pos, hole_axis)
    lowest_point_extent = peg_half_length * jnp.abs(axis_dot) + peg_radius * sin_tilt
    depth = jnp.maximum(depth_of_center + lowest_point_extent, 0.0)

    lower_end = peg_pos - peg_axis * peg_half_length * jnp.sign(axis_dot)
    rel = lower_end - hole_pos
    lower_end_depth = jnp.dot(-rel, hole_axis)
    lateral = rel - jnp.dot(rel, hole_axis) * hole_axis
    max_sin_tilt = bore_radius / jnp.maximum(peg_half_length, 1e-6)
    fits_at_this_tilt = sin_tilt <= max_sin_tilt
    contained = (
        (jnp.linalg.norm(lateral) <= bore_radius)
        & (lower_end_depth <= hole_depth)
        & fits_at_this_tilt
    )
    return jnp.where(contained, depth, 0.0)


def get_peg_hole_relative_jax(
    xpos: jnp.ndarray,
    xmat: jnp.ndarray,
    peg_body_id: int,
    hole_body_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    peg_pos = xpos[peg_body_id]
    hole_pos = xpos[hole_body_id]
    rel_pos = peg_pos - hole_pos

    peg_rot = xmat[peg_body_id].reshape(3, 3)
    hole_rot = xmat[hole_body_id].reshape(3, 3)

    peg_axis = peg_rot[:, 2]
    hole_axis = hole_rot[:, 2]

    cross = jnp.cross(peg_axis, hole_axis)
    dot = jnp.clip(jnp.dot(peg_axis, hole_axis), -1.0, 1.0)
    angle = jnp.arccos(jnp.abs(dot))
    norm = jnp.linalg.norm(cross)
    angular_error = jnp.where(norm > 1e-8, cross / norm * angle, jnp.zeros(3))

    return rel_pos, angular_error
