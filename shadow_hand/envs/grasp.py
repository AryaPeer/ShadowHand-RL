from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from shadow_hand.config import (
    DomainRandomization,
    GraspTrainConfig,
    RewardConfig,
    SceneConfig,
)
from shadow_hand.envs.vec_env import MjxVecEnv
from shadow_hand.rewards.grasp import (
    GraspRewardState,
    grasp_reward,
    init_grasp_reward_state,
)
from shadow_hand.scenes.common import (
    CUBE_GRIP_BIAS,
    CUBE_GRIP_SPAWN_XY,
    SLIDE_Z_INIT,
    apply_flexion_bias,
)
from shadow_hand.scenes.grasp import build_grasp_scene, get_object_half_height
from shadow_hand.utils.mjx_helpers import (
    get_contact_arrays,
    get_finger_object_contact_mask,
    get_fingertip_positions_jax,
    get_object_state_jax,
    get_palm_position_jax,
    pad_id_groups,
)


class GraspEnvState(NamedTuple):
    reward_state: GraspRewardState
    previous_actions: jnp.ndarray
    smoothed_actions: jnp.ndarray
    step_count: jnp.ndarray


class GraspEnv(MjxVecEnv):
    def __init__(
        self,
        num_envs: int = 2048,
        seed: int = 42,
        scene_config: SceneConfig | None = None,
        reward_config: RewardConfig | None = None,
        max_episode_steps: int = 200,
        obs_noise_std: float = 0.0,
        dr: DomainRandomization | None = None,
        p_pre_grasped: float = 0.0,
    ) -> None:
        self.scene_config = scene_config or SceneConfig()
        self.reward_config = reward_config or RewardConfig()
        self._episode_limit = max_episode_steps
        self._p_pre_grasped = jnp.array(float(p_pre_grasped))

        super().__init__(num_envs=num_envs, seed=seed, obs_noise_std=obs_noise_std, dr=dr)

        _, _, self._nm = build_grasp_scene(self.scene_config)

        init_qpos = self._mj_data.qpos.copy()
        apply_flexion_bias(init_qpos, self._mj_model)
        slide_z_jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "slide_z")
        init_qpos[self._mj_model.jnt_qposadr[slide_z_jid]] = SLIDE_Z_INIT
        self._init_qpos = jnp.array(init_qpos)

        init_qpos_grip = self._mj_data.qpos.copy()
        apply_flexion_bias(init_qpos_grip, self._mj_model, bias_map=CUBE_GRIP_BIAS)
        self._init_qpos_grip = jnp.array(init_qpos_grip)

        self._finger_touch_adr = jnp.asarray(self._nm.sensor_map.finger_touch_adr, dtype=jnp.int32)
        self._fingertip_site_ids = jnp.asarray(self._nm.fingertip_site_ids, dtype=jnp.int32)
        self._finger_geom_ids = pad_id_groups(self._nm.finger_geom_ids_per_finger)
        self._object_geom_ids = jnp.asarray([self._nm.object_geom_id], dtype=jnp.int32)

        obj_geom_id = self._nm.object_geom_id
        self._object_half_height = float(
            get_object_half_height(
                int(self._mj_model.geom_type[obj_geom_id]),
                list(self._mj_model.geom_size[obj_geom_id]),
            )
        )

    def _build_model(self) -> mujoco.MjModel:
        model, _, _ = build_grasp_scene(self.scene_config)
        return model

    def _obs_size(self) -> int:
        return 108

    def _action_size(self) -> int:
        return int(self._mj_model.nu)

    @property
    def _max_episode_steps(self) -> int:
        return self._episode_limit

    def set_curriculum_params(self, p_pre_grasped: float) -> None:
        self._p_pre_grasped = jnp.array(float(p_pre_grasped))
        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0, 0)))
        self._fused_reset = self._build_fused_reset()

    def _reset_single(
        self, mjx_model: Any, mjx_data: Any, key: jax.Array
    ) -> tuple[Any, GraspEnvState]:
        nm = self._nm
        k1, k2, k3, k4 = jax.random.split(key, 4)

        spawn_pre_grasped = jax.random.uniform(k4) < self._p_pre_grasped
        qpos = jnp.where(spawn_pre_grasped, self._init_qpos_grip, self._init_qpos)

        hand_qpos = qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        noise = jax.random.uniform(k1, shape=hand_qpos.shape, minval=-0.05, maxval=0.05)
        noise = noise.at[0:3].set(0.0)
        qpos = qpos.at[nm.hand_qpos_start : nm.hand_qpos_end].set(hand_qpos + noise)

        rand_x = jax.random.uniform(k2, minval=0.05, maxval=0.10)
        rand_y = jax.random.uniform(k3, minval=-0.03, maxval=0.03)
        obj_x = jnp.where(spawn_pre_grasped, CUBE_GRIP_SPAWN_XY[0], rand_x)
        obj_y = jnp.where(spawn_pre_grasped, CUBE_GRIP_SPAWN_XY[1], rand_y)
        obj_z = self.scene_config.table_height + self._object_half_height + 0.001

        s = nm.obj_qpos_start
        qpos = qpos.at[s : s + 3].set(jnp.array([obj_x, obj_y, obj_z]))
        qpos = qpos.at[s + 3 : s + 7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

        qvel = jnp.zeros(mjx_model.nv)

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(mjx_model, mjx_data)

        n_act = self._action_size()
        reward_state = init_grasp_reward_state(
            initial_object_height=float(obj_z),
            table_height=self.scene_config.table_height,
        )
        env_state = GraspEnvState(
            reward_state=reward_state,
            previous_actions=jnp.zeros(n_act),
            smoothed_actions=jnp.zeros(n_act),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        return mjx_data, env_state

    def _step_single(
        self,
        mjx_model: Any,
        mjx_data: Any,
        env_state: GraspEnvState,
        action: jax.Array,
    ) -> tuple[Any, GraspEnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
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
        obj_pos, obj_quat, obj_linvel, obj_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.object_body_id,
            nm.obj_qpos_start,
            nm.obj_qvel_start,
        )

        contact_geom, contact_dist = get_contact_arrays(mjx_data)
        contact_mask = get_finger_object_contact_mask(
            contact_geom,
            contact_dist,
            self._finger_geom_ids,
            self._object_geom_ids,
        )

        total, new_reward_state, info = grasp_reward(
            state=env_state.reward_state,
            finger_positions=finger_pos,
            object_position=obj_pos,
            object_linear_velocity=obj_linvel,
            finger_contact_mask=contact_mask,
            actions=smoothed,
            table_height=self.scene_config.table_height,
            cfg=self.reward_config,
        )

        fell_off = obj_pos[2] < self.scene_config.table_height - 0.05
        launched = jnp.linalg.norm(obj_pos) > 1.5
        done = fell_off | launched

        new_env_state = GraspEnvState(
            reward_state=new_reward_state,
            previous_actions=smoothed,
            smoothed_actions=smoothed,
            step_count=env_state.step_count + 1,
        )

        obs = self._get_obs_single(mjx_model, mjx_data, new_env_state)

        return mjx_data, new_env_state, obs, total, done, info

    def _get_obs_single(self, mjx_model: Any, mjx_data: Any, env_state: GraspEnvState) -> jax.Array:
        nm = self._nm

        joint_pos = mjx_data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        joint_vel = mjx_data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]

        obj_pos, obj_quat, obj_linvel, obj_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.object_body_id,
            nm.obj_qpos_start,
            nm.obj_qvel_start,
        )

        palm_pos = get_palm_position_jax(mjx_data.xpos, nm.palm_body_id)
        rel_pos = obj_pos - palm_pos
        fingertip_pos = get_fingertip_positions_jax(
            mjx_data.site_xpos, self._fingertip_site_ids
        ).flatten()

        obs = jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                obj_pos,
                obj_quat,
                obj_linvel,
                obj_angvel,
                rel_pos,
                fingertip_pos,
                env_state.previous_actions,
            ]
        )
        assert obs.shape == (self._obs_size(),), (
            f"grasp obs shape {obs.shape} != declared ({self._obs_size()},)"
        )
        return obs

    @classmethod
    def from_config(cls, config: GraspTrainConfig) -> GraspEnv:
        return cls(
            num_envs=config.num_envs,
            seed=config.seed,
            scene_config=config.scene_config,
            reward_config=config.reward_config,
            max_episode_steps=config.max_episode_steps,
            obs_noise_std=config.obs_noise_std,
            dr=config.dr,
            p_pre_grasped=(
                float(config.curriculum_stages[0][1]) if config.curriculum_stages else 0.0
            ),
        )
