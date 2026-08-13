from typing import NamedTuple

import jax.numpy as jnp

from shadow_hand.config import PegRewardConfig


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


ALIGN_GATE_CENTER = 0.02
ALIGN_GATE_K = 150.0
ENGAGE_FRAC = 0.15
ENGAGE_K = 40.0
DROP_LATERAL_MARGIN = 0.05
FORCE_PENALTY_SCALE = 0.0015
FORCE_PENALTY_CAP = 4.0
PREMATURE_RELEASE_PENALTY = -0.5
SUCCESS_ALIGN_MIN = 0.85


class PegRewardState(NamedTuple):
    was_lifted: jnp.ndarray
    was_gripped: jnp.ndarray
    insertion_hold_steps: jnp.ndarray
    initial_peg_height: jnp.ndarray
    idle_steps: jnp.ndarray


def init_peg_reward_state(initial_peg_height: float | jnp.ndarray) -> PegRewardState:
    return PegRewardState(
        was_lifted=jnp.array(False),
        was_gripped=jnp.array(False),
        insertion_hold_steps=jnp.array(0, dtype=jnp.int32),
        initial_peg_height=jnp.asarray(initial_peg_height),
        idle_steps=jnp.array(0, dtype=jnp.int32),
    )


def peg_reward(
    state: PegRewardState,
    stage: jnp.ndarray,
    finger_positions: jnp.ndarray,
    peg_position: jnp.ndarray,
    peg_axis: jnp.ndarray,
    peg_linear_velocity: jnp.ndarray,
    hole_position: jnp.ndarray,
    hole_axis: jnp.ndarray,
    insertion_depth: jnp.ndarray,
    contact_force_magnitude: jnp.ndarray,
    finger_contact_mask: jnp.ndarray,
    peg_height: jnp.ndarray,
    actions: jnp.ndarray,
    peg_length: float,
    table_height: float,
    cfg: PegRewardConfig,
) -> tuple[jnp.ndarray, PegRewardState, dict[str, jnp.ndarray]]:
    weights = cfg.weights
    ft_weights = jnp.asarray(cfg.fingertip_weights)
    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)
    released = n_contacts <= 0.5

    dists = jnp.linalg.norm(finger_positions - peg_position, axis=1)
    weighted_dist = jnp.sum(ft_weights * dists) / jnp.sum(ft_weights)
    reach = 1.0 - jnp.tanh(cfg.reach_tanh_k * weighted_dist)

    THUMB = 4
    thumb_contact = finger_contact_mask[THUMB]
    others_mask = finger_contact_mask.at[THUMB].set(False)
    others_count = jnp.sum(others_mask)

    thumb_vec = finger_positions[THUMB] - peg_position
    other_vecs = (finger_positions - peg_position) * others_mask[:, None]
    mean_other_vec = jnp.where(
        others_count > 0,
        other_vecs.sum(axis=0) / jnp.maximum(others_count, 1.0),
        jnp.zeros(3),
    )
    thumb_n = jnp.linalg.norm(thumb_vec) + 1e-6
    other_n = jnp.linalg.norm(mean_other_vec) + 1e-6
    raw_opposition = -jnp.dot(thumb_vec / thumb_n, mean_other_vec / other_n)
    opposition = jnp.where(
        thumb_contact & (others_count >= 1),
        jnp.maximum(raw_opposition, 0.0),
        0.0,
    )

    contact_scale = jnp.minimum(n_contacts / 3.0, 1.0)
    tripod_bonus = 0.5 * (thumb_contact & (others_count >= 2)).astype(jnp.float32)
    grasp = contact_scale * (0.3 + 0.7 * opposition) + tripod_bonus

    lift_height = jnp.maximum(peg_height - state.initial_peg_height, 0.0)
    lift_gate = (n_contacts >= 2).astype(jnp.float32)
    lift = jnp.minimum(lift_height / cfg.lift_target, 1.0) * lift_gate

    obj_speed = jnp.linalg.norm(peg_linear_velocity)
    height_gate = _sigmoid(cfg.hold_height_smoothness_k * (lift_height - cfg.lift_target))
    speed_gate = _sigmoid(
        cfg.hold_velocity_smoothness_k * (cfg.hold_velocity_threshold - obj_speed)
    )
    holding = height_gate * speed_gate * contact_scale

    was_lifted_next = state.was_lifted | (lift_height >= cfg.lift_target)

    axis_align = jnp.abs(jnp.dot(peg_axis, hole_axis))
    axis_in_grip = axis_align * contact_scale

    lateral_dist = jnp.linalg.norm(peg_position[:2] - hole_position[:2])
    lateral_factor = 1.0 - jnp.tanh(cfg.lateral_gate_k * lateral_dist)
    peg_clearance = jnp.maximum(peg_height - table_height - peg_length * 0.5, 0.0)
    align_weight = _sigmoid((peg_clearance - ALIGN_GATE_CENTER) * ALIGN_GATE_K)

    insertion_fraction = jnp.clip(insertion_depth / peg_length, 0.0, 1.0)
    engaged = _sigmoid(ENGAGE_K * (insertion_fraction - ENGAGE_FRAC))
    released_f = released.astype(jnp.float32)

    align_presence = jnp.maximum(contact_scale, engaged * released_f)
    align = axis_align * lateral_factor * align_weight * align_presence

    depth_reward = cfg.depth_reward_scale * insertion_fraction * lateral_factor * released_f

    axis_dot_ph = jnp.dot(peg_axis, hole_axis)
    axis_sign = jnp.where(axis_dot_ph >= 0.0, 1.0, -1.0)
    tip = peg_position - peg_axis * axis_sign * (peg_length / 2.0)
    top = peg_position + peg_axis * axis_sign * (peg_length / 2.0)
    target_tip = hole_position + hole_axis * cfg.release_height
    target_top = target_tip + hole_axis * peg_length
    keypoint_dist = jnp.linalg.norm(tip - target_tip) + jnp.linalg.norm(top - target_top)
    carry_gate = ((lift_height >= cfg.carry_clear_height) & (n_contacts >= 2)).astype(jnp.float32)
    place_gate = jnp.maximum(carry_gate, jnp.clip(insertion_fraction / 0.05, 0.0, 1.0))
    place = (1.0 - jnp.tanh(cfg.place_k * keypoint_dist)) * place_gate

    place_release = engaged * released_f

    new_hold = jnp.where(
        (insertion_fraction > cfg.success_threshold)
        & released
        & (axis_align > SUCCESS_ALIGN_MIN)
        & (obj_speed < cfg.hold_velocity_threshold),
        state.insertion_hold_steps + 1,
        jnp.array(0, dtype=jnp.int32),
    )
    is_success = new_hold >= cfg.peg_hold_steps
    success = jnp.where(is_success, cfg.success_bonus_per_step, 0.0)

    force_excess = jnp.maximum(0.0, contact_force_magnitude - cfg.force_threshold)
    force_penalty = jnp.maximum(-FORCE_PENALTY_CAP, -FORCE_PENALTY_SCALE * force_excess)

    just_dropped = (
        state.was_lifted
        & (lift_height < 0.01)
        & (insertion_fraction < 0.1)
        & (lateral_dist > DROP_LATERAL_MARGIN)
    )
    drop = jnp.where(just_dropped, cfg.drop_penalty, 0.0)
    was_lifted = jnp.where(just_dropped, False, was_lifted_next)

    was_gripped_next = state.was_gripped | (n_contacts >= 2)
    premature_release = (
        state.was_gripped
        & released
        & (insertion_fraction < ENGAGE_FRAC)
        & (lateral_dist > DROP_LATERAL_MARGIN)
    )
    premature_penalty = jnp.where(premature_release, PREMATURE_RELEASE_PENALTY, 0.0)

    action_penalty = -cfg.action_penalty_scale * jnp.sum(actions**2)

    idle_active = (n_contacts == 0) & (place < 0.3)
    new_idle_steps = jnp.where(idle_active, state.idle_steps + 1, jnp.array(0, dtype=jnp.int32))
    idle_raw = jnp.where(new_idle_steps >= cfg.idle_grace_steps, cfg.no_contact_idle_penalty, 0.0)
    idle_penalty = weights.idle * idle_raw

    total = (
        weights.reach * reach
        + weights.grasp * grasp
        + weights.opposition * opposition
        + weights.axis_in_grip * axis_in_grip
        + weights.lift * lift
        + weights.holding * holding
        + weights.align * align
        + weights.place * place
        + weights.depth * depth_reward
        + weights.place_release * place_release
        + weights.success * success
        + weights.force * force_penalty
        + weights.drop * drop
        + premature_penalty
        + weights.action_penalty * action_penalty
        + idle_penalty
    )

    new_state = PegRewardState(
        was_lifted=was_lifted,
        was_gripped=was_gripped_next,
        insertion_hold_steps=new_hold,
        initial_peg_height=state.initial_peg_height,
        idle_steps=new_idle_steps,
    )

    info = {
        "reward/reach": reach,
        "reward/grasp": grasp,
        "reward/grasp_quality": opposition,
        "reward/axis_in_grip": axis_in_grip,
        "reward/lift": lift,
        "reward/holding": holding,
        "reward/align": align,
        "reward/place": place,
        "reward/depth": depth_reward,
        "reward/place_release": place_release,
        "reward/success": success,
        "reward/force_penalty": force_penalty,
        "reward/drop": drop,
        "reward/premature_release": premature_penalty,
        "reward/action_penalty": action_penalty,
        "reward/idle_penalty": idle_raw,
        "reward/total": total,
        "metrics/keypoint_dist": keypoint_dist,
        "metrics/stage": stage.astype(jnp.float32),
        "metrics/num_finger_contacts": n_contacts,
        "metrics/peg_height": peg_height,
        "metrics/insertion_depth": insertion_depth,
        "metrics/contact_force": contact_force_magnitude,
        "metrics/lateral_distance": lateral_dist,
        "metrics/insertion_hold_steps": new_hold.astype(jnp.float32),
        "metrics/axis_align": axis_align,
    }

    return total, new_state, info
