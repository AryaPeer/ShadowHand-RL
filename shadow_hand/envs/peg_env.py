from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from shadow_hand.config import (
    DomainRandomization,
    MjxPegTrainConfig,
    PegRewardConfig,
    PegSceneConfig,
)
from shadow_hand.envs.mjx_vec_env import MjxVecEnv
from shadow_hand.envs.peg_scene_builder import (
    PEG_SLIDE_Z_RANGE,
    WALL_SENSOR_NAMES,
    build_peg_scene,
)
from shadow_hand.envs.scene_builder import (
    GRIP_BIAS,
    apply_flexion_bias,
    build_grip_ctrl,
)
from shadow_hand.rewards.peg_reward import (
    PegRewardState,
    init_peg_reward_state,
    peg_reward,
)
from shadow_hand.utils.mjx_helpers import (
    get_body_axis_jax,
    get_contact_arrays,
    get_finger_object_contact_mask,
    get_finger_touch_from_sensors,
    get_fingertip_positions_jax,
    get_insertion_depth_jax,
    get_object_state_jax,
    get_palm_position_jax,
    get_peg_hole_relative_jax,
    pad_id_groups,
)


class PegEnvState(NamedTuple):
    reward_state: PegRewardState
    previous_actions: jnp.ndarray
    smoothed_actions: jnp.ndarray
    stage: jnp.ndarray
    no_contact_grace: jnp.ndarray
    step_count: jnp.ndarray
    key: jax.Array


class ShadowHandPegMjxEnv(MjxVecEnv):
    def __init__(
        self,
        num_envs: int = 2048,
        seed: int = 42,
        scene_config: PegSceneConfig | None = None,
        reward_config: PegRewardConfig | None = None,
        max_episode_steps: int = 500,
        obs_noise_std: float = 0.0,
        dr: DomainRandomization | None = None,
        carry_floor: float = 0.0,
    ) -> None:
        self.scene_config = scene_config or PegSceneConfig()
        self.reward_config = reward_config or PegRewardConfig()
        if self.scene_config.spawn_max_radius <= self.scene_config.spawn_min_radius:
            raise ValueError(
                f"spawn_max_radius ({self.scene_config.spawn_max_radius}) must exceed "
                f"spawn_min_radius ({self.scene_config.spawn_min_radius})"
            )

        self._episode_limit = max_episode_steps

        self._carry_floor = jnp.array(float(carry_floor))

        super().__init__(num_envs=num_envs, seed=seed, obs_noise_std=obs_noise_std, dr=dr)

        self._rebuild_peg_caches()

    def _build_model(self) -> mujoco.MjModel:
        model, _, _ = build_peg_scene(self.scene_config)
        return model

    def _obs_size(self) -> int:
        return 128 + len(WALL_SENSOR_NAMES) + 1

    def _action_size(self) -> int:
        return int(self._mj_model.nu)

    @property
    def _max_episode_steps(self) -> int:
        return self._episode_limit

    def _rebuild_peg_caches(self) -> None:
        self._reset_cache: dict[float, tuple] = {}
        _, _, self._nm = build_peg_scene(self.scene_config)

        init_qpos_grip = self._mj_data.qpos.copy()
        apply_flexion_bias(init_qpos_grip, self._mj_model, bias_map=GRIP_BIAS)
        self._init_qpos_grip = jnp.array(init_qpos_grip)

        self._grip_ctrl = jnp.array(build_grip_ctrl(self._mj_model))

        self._grasp_site_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )

        self._finger_touch_adr = jnp.asarray(self._nm.sensor_map.finger_touch_adr, dtype=jnp.int32)
        self._wall_force_adr = jnp.asarray(self._nm.sensor_map.wall_force_adr, dtype=jnp.int32)
        self._fingertip_site_ids = jnp.asarray(self._nm.fingertip_site_ids, dtype=jnp.int32)
        self._finger_geom_ids = pad_id_groups(self._nm.finger_geom_ids_per_finger)
        self._peg_geom_ids = jnp.asarray([self._nm.peg_geom_id], dtype=jnp.int32)
        self._peg_length = self.scene_config.peg_half_length * 2.0
        self._bore_radius = self.scene_config.peg_radius + self.scene_config.clearance
        self._slide_qadr = jnp.asarray(
            [self._nm.hand_qpos_start + i for i in range(3)], dtype=jnp.int32
        )
        self._slide_act_adr = jnp.asarray([0, 1, 2], dtype=jnp.int32)
        self._slide_lo = jnp.array([-0.15, -0.15, PEG_SLIDE_Z_RANGE[0]])
        self._slide_hi = jnp.array([0.15, 0.15, PEG_SLIDE_Z_RANGE[1]])
        self._engaged_center_z = (
            self.scene_config.table_height
            + self.scene_config.hole_top_above_table
            + 0.01
            + self.scene_config.peg_half_length
        )

    def set_curriculum_params(self, clearance: float, carry_floor: float = 0.0) -> None:
        self._carry_floor = jnp.array(float(carry_floor))

        clearance_f = float(clearance)
        clearance_changed = abs(clearance_f - float(self.scene_config.clearance)) > 1e-9

        if clearance_changed:
            self.scene_config.clearance = clearance_f
            self._mj_model = self._build_model()
            self._mj_data = mujoco.MjData(self._mj_model)
            self._mjx_model = mjx.put_model(self._mj_model)

            n_act = self._action_size()
            self._ctrl_low = jnp.array(self._mj_model.actuator_ctrlrange[:n_act, 0])
            self._ctrl_high = jnp.array(self._mj_model.actuator_ctrlrange[:n_act, 1])

            self._rebuild_peg_caches()
            self._batched_get_obs = jax.jit(jax.vmap(self._get_obs_single, in_axes=(None, 0, 0)))
            self._fused_step = self._build_fused_step()

        key = round(float(carry_floor), 6)
        cached = self._reset_cache.get(key)
        if cached is not None:
            self._batched_reset, self._fused_reset = cached
        else:
            self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0, 0)))
            self._fused_reset = self._build_fused_reset()
            self._reset_cache[key] = (self._batched_reset, self._fused_reset)

        if clearance_changed and self._mjx_data_batch is not None:
            self.reset()

    def _reset_single(
        self, mjx_model: Any, mjx_data: Any, key: jax.Array
    ) -> tuple[Any, PegEnvState]:
        nm = self._nm
        k_noise, k_r, k_theta, k_t, k_state = jax.random.split(key, 5)

        qpos = self._init_qpos_grip

        hand_qpos = qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        noise = jax.random.uniform(k_noise, shape=hand_qpos.shape, minval=-0.05, maxval=0.05)
        noise = noise.at[0:3].set(0.0)
        qpos = qpos.at[nm.hand_qpos_start : nm.hand_qpos_end].set(hand_qpos + noise)
        qvel = jnp.zeros(mjx_model.nv)

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.kinematics(mjx_model, mjx_data)

        hole_x = float(self.scene_config.hole_offset[0])
        hole_y = float(self.scene_config.hole_offset[1])
        pick_x = float(self.scene_config.pick_center[0])
        pick_y = float(self.scene_config.pick_center[1])
        min_r = float(self.scene_config.spawn_min_radius)
        max_r = float(self.scene_config.spawn_max_radius)
        r = jax.random.uniform(k_r, minval=min_r, maxval=max_r)
        theta = jax.random.uniform(k_theta, minval=-math.pi, maxval=math.pi)
        table_peg_z = self.scene_config.table_height + self.scene_config.peg_half_length + 0.001
        carry_start = jnp.array(
            [pick_x + r * jnp.cos(theta), pick_y + r * jnp.sin(theta), table_peg_z]
        )

        t = self._carry_floor + jax.random.uniform(k_t) * (1.0 - self._carry_floor)
        target_engaged = jnp.array([hole_x, hole_y, self._engaged_center_z])
        grip_target = target_engaged + t * (carry_start - target_engaged)
        grasp_p0 = mjx_data.site_xpos[self._grasp_site_id]
        slide_delta = jnp.clip(grip_target - grasp_p0, self._slide_lo, self._slide_hi)
        qpos = qpos.at[self._slide_qadr].add(slide_delta)
        mjx_data = mjx_data.replace(qpos=qpos)
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        peg_xyz = mjx_data.site_xpos[self._grasp_site_id]

        s = nm.peg_qpos_start
        qpos = mjx_data.qpos.at[s : s + 3].set(peg_xyz)
        qpos = qpos.at[s + 3 : s + 7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

        mjx_data = mjx_data.replace(qpos=qpos, qvel=jnp.zeros(mjx_model.nv))

        settle_ctrl = self._grip_ctrl.at[self._slide_act_adr].set(qpos[self._slide_qadr])
        init_action = jnp.clip(
            2.0 * (settle_ctrl - self._ctrl_low) / (self._ctrl_high - self._ctrl_low) - 1.0,
            -1.0,
            1.0,
        )
        mjx_data = mjx_data.replace(ctrl=settle_ctrl)

        def _settle(data: Any, _: Any) -> tuple[Any, None]:
            return mjx.step(mjx_model, data), None

        mjx_data, _ = jax.lax.scan(_settle, mjx_data, None, length=5)

        table_spawn_height = (
            self.scene_config.table_height + self.scene_config.peg_half_length + 0.001
        )
        initial_peg_height = jnp.minimum(mjx_data.xpos[nm.peg_body_id][2], table_spawn_height)

        env_state = PegEnvState(
            reward_state=init_peg_reward_state(initial_peg_height),
            previous_actions=init_action,
            smoothed_actions=init_action,
            stage=jnp.array(0, dtype=jnp.int32),
            no_contact_grace=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            key=k_state,
        )

        return mjx_data, env_state

    def _step_single(
        self,
        mjx_model: Any,
        mjx_data: Any,
        env_state: PegEnvState,
        action: jax.Array,
    ) -> tuple[Any, PegEnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        nm = self._nm
        action = jnp.clip(action, -1.0, 1.0)

        alpha = self.scene_config.action_smoothing_alpha
        smoothed = (1.0 - alpha) * env_state.smoothed_actions + alpha * action

        ctrl = self._ctrl_low + (smoothed + 1.0) / 2.0 * (self._ctrl_high - self._ctrl_low)
        mjx_data = mjx_data.replace(ctrl=ctrl)

        def _substep(data: Any, _: Any) -> tuple[Any, None]:
            return mjx.step(mjx_model, data), None

        mjx_data, _ = jax.lax.scan(_substep, mjx_data, None, length=self.scene_config.frame_skip)

        finger_pos = get_fingertip_positions_jax(mjx_data.site_xpos, self._fingertip_site_ids)
        peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.peg_body_id,
            nm.peg_qpos_start,
            nm.peg_qvel_start,
        )

        touch_vals, _ = get_finger_touch_from_sensors(mjx_data.sensordata, self._finger_touch_adr)
        contact_geom, contact_dist = get_contact_arrays(mjx_data)
        contact_mask = get_finger_object_contact_mask(
            contact_geom,
            contact_dist,
            self._finger_geom_ids,
            self._peg_geom_ids,
        )
        n_contacts = jnp.sum(contact_mask).astype(jnp.float32)

        peg_axis = get_body_axis_jax(mjx_data.xmat, nm.peg_body_id)
        hole_axis = get_body_axis_jax(mjx_data.xmat, nm.hole_body_id)
        hole_pos = mjx_data.xpos[nm.hole_body_id]

        peg_half_length = self.scene_config.peg_half_length
        peg_radius = self.scene_config.peg_radius
        insertion_depth = get_insertion_depth_jax(
            mjx_data.xpos,
            mjx_data.xmat,
            nm.peg_body_id,
            nm.hole_body_id,
            peg_half_length,
            peg_radius,
            self._bore_radius,
            self.scene_config.hole_depth,
        )

        wall_vals = mjx_data.sensordata[self._wall_force_adr]
        contact_force_mag = jnp.sum(wall_vals)

        fingers_on_peg = n_contacts >= 2
        peg_lifted = peg_pos[2] > env_state.reward_state.initial_peg_height + 0.02
        peg_near_hole = jnp.linalg.norm(peg_pos[:2] - hole_pos[:2]) < 0.03
        peg_aligned = jnp.abs(jnp.dot(peg_axis, hole_axis)) > 0.95
        peg_inserted = insertion_depth > 0.02

        new_grace = jnp.where(
            n_contacts == 0, env_state.no_contact_grace + 1, jnp.array(0, dtype=jnp.int32)
        )

        target = jnp.array(0, dtype=jnp.int32)
        target = jnp.where(fingers_on_peg, 1, target)
        target = jnp.where(peg_lifted, 2, target)
        target = jnp.where((peg_near_hole & peg_aligned) | peg_inserted, 3, target)

        new_stage = jnp.where(
            (new_grace >= 5) & ~peg_inserted, 0, jnp.maximum(env_state.stage, target)
        )

        peg_height = peg_pos[2]

        total, new_reward_state, info = peg_reward(
            state=env_state.reward_state,
            stage=new_stage,
            finger_positions=finger_pos,
            peg_position=peg_pos,
            peg_axis=peg_axis,
            peg_linear_velocity=peg_linvel,
            hole_position=hole_pos,
            hole_axis=hole_axis,
            insertion_depth=insertion_depth,
            contact_force_magnitude=contact_force_mag,
            finger_contact_mask=contact_mask,
            peg_height=peg_height,
            actions=smoothed,
            peg_length=self._peg_length,
            table_height=self.scene_config.table_height,
            cfg=self.reward_config,
        )

        insertion_complete = (
            (insertion_depth > self.reward_config.success_threshold * self._peg_length)
            & (new_reward_state.insertion_hold_steps >= self.reward_config.peg_hold_steps)
            & (n_contacts <= 0.5)
        )
        fell = peg_pos[2] < self.scene_config.table_height - 0.1
        done = fell
        info["is_success"] = insertion_complete.astype(jnp.float32)

        new_env_state = PegEnvState(
            reward_state=new_reward_state,
            previous_actions=smoothed,
            smoothed_actions=smoothed,
            stage=new_stage,
            no_contact_grace=new_grace,
            step_count=env_state.step_count + 1,
            key=env_state.key,
        )

        obs = self._get_obs_single(mjx_model, mjx_data, new_env_state)

        return mjx_data, new_env_state, obs, total, done, info

    def _get_obs_single(self, mjx_model: Any, mjx_data: Any, env_state: PegEnvState) -> jax.Array:
        nm = self._nm

        joint_pos = mjx_data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        joint_vel = mjx_data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]

        peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.peg_body_id,
            nm.peg_qpos_start,
            nm.peg_qvel_start,
        )

        hole_pos = mjx_data.xpos[nm.hole_body_id]
        hole_quat = mjx_data.xquat[nm.hole_body_id]

        rel_pos, ang_error = get_peg_hole_relative_jax(
            mjx_data.xpos, mjx_data.xmat, nm.peg_body_id, nm.hole_body_id
        )

        finger_pos = get_fingertip_positions_jax(mjx_data.site_xpos, self._fingertip_site_ids)
        fingertip_peg_dist = jnp.linalg.norm(finger_pos - peg_pos, axis=1)

        palm_pos = get_palm_position_jax(mjx_data.xpos, nm.palm_body_id)
        rel_peg_to_palm = peg_pos - palm_pos

        insertion_depth = get_insertion_depth_jax(
            mjx_data.xpos,
            mjx_data.xmat,
            nm.peg_body_id,
            nm.hole_body_id,
            self.scene_config.peg_half_length,
            self.scene_config.peg_radius,
            self._bore_radius,
            self.scene_config.hole_depth,
        )

        per_wall_forces = mjx_data.sensordata[self._wall_force_adr]
        contact_force_mag = jnp.sum(per_wall_forces)
        contact_forces = jnp.concatenate([per_wall_forces, jnp.array([contact_force_mag])])

        obs = jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                peg_pos,
                peg_quat,
                peg_linvel,
                peg_angvel,
                hole_pos,
                hole_quat,
                rel_pos,
                ang_error,
                finger_pos.flatten(),
                fingertip_peg_dist,
                rel_peg_to_palm,
                jnp.array([insertion_depth]),
                contact_forces,
                jnp.asarray(env_state.stage, dtype=jnp.float32).reshape(1),
                env_state.previous_actions,
            ]
        )
        assert obs.shape == (self._obs_size(),), (
            f"peg obs shape {obs.shape} != declared ({self._obs_size()},)"
        )
        return obs

    @classmethod
    def from_config(cls, config: MjxPegTrainConfig) -> ShadowHandPegMjxEnv:
        return cls(
            num_envs=config.num_envs,
            seed=config.seed,
            scene_config=config.scene_config,
            reward_config=config.reward_config,
            max_episode_steps=config.max_episode_steps,
            obs_noise_std=config.obs_noise_std,
            dr=config.dr,
            carry_floor=config.adaptive_curriculum.carry_floor_levels[0],
        )
