from typing import NamedTuple

import jax.numpy as jnp

from shadow_hand.config import PickPlaceRewardConfig


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


class PickPlaceRewardState(NamedTuple):
    was_lifted: jnp.ndarray
    initial_height_above_table: jnp.ndarray
    idle_steps: jnp.ndarray
    place_hold_counter: jnp.ndarray


def init_pickplace_reward_state(
    initial_object_height: float,
    table_height: float,
) -> PickPlaceRewardState:
    return PickPlaceRewardState(
        was_lifted=jnp.array(False),
        initial_height_above_table=jnp.maximum(
            jnp.array(initial_object_height) - jnp.array(table_height), 0.0
        ),
        idle_steps=jnp.array(0, dtype=jnp.int32),
        place_hold_counter=jnp.array(0, dtype=jnp.int32),
    )


def pickplace_reward(
    state: PickPlaceRewardState,
    finger_positions: jnp.ndarray,
    palm_position: jnp.ndarray,
    object_position: jnp.ndarray,
    object_linear_velocity: jnp.ndarray,
    finger_contact_mask: jnp.ndarray,
    goal_xy: jnp.ndarray,
    actions: jnp.ndarray,
    table_height: float,
    object_half_extent: float,
    cfg: PickPlaceRewardConfig,
) -> tuple[jnp.ndarray, PickPlaceRewardState, dict[str, jnp.ndarray]]:
    weights = cfg.weights
    ft_weights = jnp.asarray(cfg.fingertip_weights)

    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)
    obj_height = object_position[2]
    height_above_table = obj_height - table_height
    lift_height = jnp.maximum(height_above_table - state.initial_height_above_table, 0.0)

    dists = jnp.linalg.norm(finger_positions - object_position, axis=1)
    weighted_dist = jnp.sum(ft_weights * dists) / jnp.sum(ft_weights)
    reaching = 1.0 - jnp.tanh(cfg.reach_tanh_k * weighted_dist)

    finger_z_below = finger_positions[:, 2] <= (object_position[2] + 0.015)
    side_count = jnp.sum(finger_contact_mask & finger_z_below).astype(jnp.float32)
    side_ratio = jnp.where(n_contacts > 0, side_count / jnp.maximum(n_contacts, 1.0), 0.0)

    contact_scale = jnp.tanh(n_contacts / 2.0)
    grasping = contact_scale * (0.3 + 0.7 * side_ratio)

    lift_gate = (n_contacts >= 2).astype(jnp.float32)
    lifting = jnp.clip(lift_height / cfg.lift_target, 0.0, 1.0) * lift_gate

    obj_speed = jnp.linalg.norm(object_linear_velocity)
    height_gate = _sigmoid(cfg.hold_height_smoothness_k * (lift_height - cfg.lift_target))
    speed_gate = _sigmoid(
        cfg.hold_velocity_smoothness_k * (cfg.hold_velocity_threshold - obj_speed)
    )
    holding = height_gate * speed_gate * contact_scale

    xy_dist = jnp.linalg.norm(object_position[:2] - goal_xy)
    carry_gate = ((lift_height >= cfg.carry_clear_height) & (n_contacts >= 2)).astype(jnp.float32)
    transport = (1.0 - jnp.tanh(cfg.transport_tanh_k * xy_dist)) * carry_gate

    palm_to_goal = jnp.linalg.norm(palm_position[:2] - goal_xy)
    hand_to_goal = (1.0 - jnp.tanh(cfg.hand_to_goal_tanh_k * palm_to_goal)) * carry_gate

    rest_z = table_height + object_half_extent
    height_err = jnp.abs(obj_height - rest_z)
    on_table = _sigmoid(cfg.on_table_k * (cfg.on_table_tol - height_err))
    at_rest = _sigmoid(cfg.at_rest_k * (cfg.hold_velocity_threshold - obj_speed))
    at_goal_xy = 1.0 - jnp.tanh(cfg.goal_tanh_k * xy_dist)

    was_lifted_next = state.was_lifted | (lift_height >= cfg.drop_arm_height)
    picked_gate = was_lifted_next.astype(jnp.float32)
    placed = at_goal_xy * on_table * at_rest * picked_gate

    released = n_contacts <= 0.5
    place_release = placed * released.astype(jnp.float32)
    at_goal_success = (
        (xy_dist < cfg.goal_radius)
        & (height_err < cfg.on_table_tol)
        & (obj_speed < cfg.hold_velocity_threshold)
        & released
        & was_lifted_next
    )
    new_place_hold = jnp.where(
        at_goal_success, state.place_hold_counter + 1, jnp.array(0, dtype=jnp.int32)
    )
    is_success = new_place_hold >= cfg.place_hold_steps
    success = jnp.where(is_success, cfg.success_bonus_per_step, 0.0)

    just_dropped = state.was_lifted & (lift_height < 0.01) & (xy_dist > cfg.goal_radius + 0.02)
    drop = jnp.where(just_dropped, cfg.drop_penalty, 0.0)
    was_lifted = jnp.where(just_dropped, False, was_lifted_next)

    idle_active = (n_contacts == 0) & (placed < 0.3)
    new_idle_steps = jnp.where(idle_active, state.idle_steps + 1, jnp.array(0, dtype=jnp.int32))
    idle_raw = jnp.where(new_idle_steps >= cfg.idle_grace_steps, cfg.no_contact_idle_penalty, 0.0)
    idle_penalty = weights.idle * idle_raw

    action_penalty = -cfg.action_penalty_scale * jnp.sum(actions**2)

    total = (
        weights.reaching * reaching
        + weights.grasping * grasping
        + weights.lifting * lifting
        + weights.holding * holding
        + weights.transport * transport
        + weights.hand_to_goal * hand_to_goal
        + weights.placed * placed
        + weights.release * place_release
        + weights.success * success
        + weights.drop * drop
        + weights.action_penalty * action_penalty
        + idle_penalty
    )

    new_state = PickPlaceRewardState(
        was_lifted=was_lifted,
        initial_height_above_table=state.initial_height_above_table,
        idle_steps=new_idle_steps,
        place_hold_counter=new_place_hold,
    )

    info = {
        "reward/reaching": reaching,
        "reward/grasping": grasping,
        "reward/grasp_quality": side_ratio,
        "reward/lifting": lifting,
        "reward/holding": holding,
        "reward/transport": transport,
        "reward/hand_to_goal": hand_to_goal,
        "reward/placed": placed,
        "reward/place_release": place_release,
        "reward/success": success,
        "reward/drop": drop,
        "reward/idle_penalty": idle_raw,
        "reward/action_penalty": action_penalty,
        "reward/total": total,
        "metrics/num_finger_contacts": n_contacts,
        "metrics/object_height": obj_height,
        "metrics/object_speed": obj_speed,
        "metrics/xy_dist_to_goal": xy_dist,
        "metrics/mean_fingertip_dist": jnp.mean(dists),
        "metrics/place_hold_steps": new_place_hold.astype(jnp.float32),
        "is_success": is_success.astype(jnp.float32),
    }

    return total, new_state, info
