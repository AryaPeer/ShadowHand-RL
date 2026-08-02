from typing import NamedTuple

import jax.numpy as jnp

from dexterous_hand.config import RewardConfig


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


class GraspRewardState(NamedTuple):
    was_lifted: jnp.ndarray
    initial_height_above_table: jnp.ndarray
    idle_steps: jnp.ndarray
    success_hold_counter: jnp.ndarray


def init_grasp_reward_state(
    initial_object_height: float,
    table_height: float,
) -> GraspRewardState:
    return GraspRewardState(
        was_lifted=jnp.array(False),
        initial_height_above_table=jnp.maximum(
            jnp.array(initial_object_height) - jnp.array(table_height), 0.0
        ),
        idle_steps=jnp.array(0, dtype=jnp.int32),
        success_hold_counter=jnp.array(0, dtype=jnp.int32),
    )


def grasp_reward(
    state: GraspRewardState,
    finger_positions: jnp.ndarray,
    object_position: jnp.ndarray,
    object_linear_velocity: jnp.ndarray,
    finger_contact_mask: jnp.ndarray,
    actions: jnp.ndarray,
    table_height: float,
    cfg: RewardConfig,
) -> tuple[jnp.ndarray, GraspRewardState, dict[str, jnp.ndarray]]:
    weights = cfg.weights
    lift_target = cfg.lift_target
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
    lifting = jnp.clip(lift_height / lift_target, 0.0, 1.0) * lift_gate

    obj_speed = jnp.linalg.norm(object_linear_velocity)
    height_gate = _sigmoid(cfg.hold_height_smoothness_k * (lift_height - lift_target))
    speed_gate = _sigmoid(
        cfg.hold_velocity_smoothness_k * (cfg.hold_velocity_threshold - obj_speed)
    )
    holding = height_gate * speed_gate * contact_scale

    was_lifted_next = state.was_lifted | (lift_height >= cfg.drop_arm_height)

    just_dropped = state.was_lifted & (lift_height < 0.01)
    drop = jnp.where(just_dropped, cfg.drop_penalty, 0.0)
    was_lifted = jnp.where(just_dropped, False, was_lifted_next)

    lift_factor = jnp.clip(lift_height / lift_target, 0.0, 1.0)
    contact_factor = jnp.clip(n_contacts / 3.0, 0.0, 1.0)
    speed_factor = _sigmoid(20.0 * (0.2 - obj_speed))
    at_target = (lift_factor * contact_factor * speed_factor) >= 0.85
    new_success_hold = jnp.where(
        at_target, state.success_hold_counter + 1, jnp.array(0, dtype=jnp.int32)
    )
    is_success = new_success_hold >= cfg.success_hold_steps
    success = jnp.where(is_success, cfg.success_bonus_per_step, 0.0)

    idle_active = n_contacts == 0
    new_idle_steps = jnp.where(idle_active, state.idle_steps + 1, jnp.array(0, dtype=jnp.int32))
    idle_raw = jnp.where(new_idle_steps >= cfg.idle_grace_steps, cfg.no_contact_idle_penalty, 0.0)
    idle_penalty = weights.idle * idle_raw

    action_penalty = -cfg.action_penalty_scale * jnp.sum(actions**2)

    total = (
        weights.reaching * reaching
        + weights.grasping * grasping
        + weights.lifting * lifting
        + weights.holding * holding
        + weights.drop * drop
        + weights.success * success
        + weights.action_penalty * action_penalty
        + idle_penalty
    )

    new_state = GraspRewardState(
        was_lifted=was_lifted,
        initial_height_above_table=state.initial_height_above_table,
        idle_steps=new_idle_steps,
        success_hold_counter=new_success_hold,
    )

    info = {
        "reward/reaching": reaching,
        "reward/grasping": grasping,
        "reward/grasp_quality": side_ratio,
        "reward/lifting": lifting,
        "reward/holding": holding,
        "reward/drop": drop,
        "reward/success": success,
        "reward/idle_penalty": idle_raw,
        "reward/action_penalty": action_penalty,
        "reward/total": total,
        "metrics/num_finger_contacts": n_contacts,
        "metrics/object_height": obj_height,
        "metrics/object_speed": obj_speed,
        "metrics/mean_fingertip_dist": jnp.mean(dists),
        "metrics/success_hold_steps": new_success_hold.astype(jnp.float32),
        "is_success": is_success.astype(jnp.float32),
    }

    return total, new_state, info
