from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from shadow_hand.config import DomainRandomization


class DRParams(NamedTuple):
    mass_mult: jnp.ndarray
    friction_mult: jnp.ndarray
    gain_mult: jnp.ndarray


def _apply_dr(mjx_model: Any, dr_params: DRParams) -> Any:
    return mjx_model.replace(
        body_mass=mjx_model.body_mass * dr_params.mass_mult,
        geom_friction=mjx_model.geom_friction.at[:, 0].multiply(dr_params.friction_mult),
        actuator_gainprm=mjx_model.actuator_gainprm.at[:, 0].multiply(dr_params.gain_mult),
    )


def _mask(needs_reset: jnp.ndarray, leaf: jnp.ndarray) -> jnp.ndarray:
    if leaf.ndim > 1:
        return needs_reset.reshape(needs_reset.shape + (1,) * (leaf.ndim - 1))
    return needs_reset


def _merge(needs_reset: jnp.ndarray, reset_tree: Any, old_tree: Any) -> Any:
    return jax.tree.map(lambda r, n: jnp.where(_mask(needs_reset, r), r, n), reset_tree, old_tree)


class MjxVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        seed: int = 42,
        obs_noise_std: float = 0.0,
        dr: DomainRandomization | None = None,
    ) -> None:

        self._num_envs = num_envs
        self._obs_noise_std = float(obs_noise_std)
        self._dr_config = dr if dr is not None else DomainRandomization(enabled=False)

        self._mj_model = self._build_model()
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model)

        n_obs = self._obs_size()
        n_act = self._action_size()

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        super().__init__(num_envs, observation_space, action_space)

        self._master_key = jax.random.PRNGKey(seed)
        noise_key, self._master_key = jax.random.split(self._master_key)
        self._obs_noise_key = jax.device_put(noise_key)
        dr_key, self._master_key = jax.random.split(self._master_key)
        self._dr_key = jax.device_put(dr_key)
        self._env_keys = jax.device_put(jax.random.split(self._master_key, num_envs))

        self._ctrl_low = jnp.array(self._mj_model.actuator_ctrlrange[:n_act, 0])
        self._ctrl_high = jnp.array(self._mj_model.actuator_ctrlrange[:n_act, 1])

        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0, 0)))
        self._batched_get_obs = jax.jit(jax.vmap(self._get_obs_single, in_axes=(None, 0, 0)))
        self._fused_step = self._build_fused_step()
        self._fused_reset = self._build_fused_reset()

        self._mjx_data_batch = None
        self._env_state_batch = None
        self._step_count = jax.device_put(jnp.zeros(num_envs, dtype=jnp.int32))
        self._dr_params_batch: DRParams | None = None

        self._pending_obs: np.ndarray | None = None
        self._pending_rewards: np.ndarray | None = None
        self._pending_dones: np.ndarray | None = None
        self._pending_infos: list[dict] | None = None
        self._full_reset_count = 0

    def _build_fused_step(self) -> Any:
        dr_enabled = self._dr_config.enabled
        noise_std = self._obs_noise_std
        max_steps = self._max_episode_steps

        def _dr_step(
            model: Any, dr_params: DRParams | None, data: Any, state: Any, action: jax.Array
        ) -> Any:
            local_model = model if dr_params is None else _apply_dr(model, dr_params)
            return self._step_single(local_model, data, state, action)

        vstep = jax.vmap(_dr_step, in_axes=(None, 0 if dr_enabled else None, 0, 0, 0))

        def _fused(
            model: Any,
            dr_params: DRParams | None,
            data: Any,
            state: Any,
            actions: jax.Array,
            step_count: jax.Array,
            noise_key: jax.Array,
        ) -> Any:
            new_data, new_state, obs, rewards, dones, info = vstep(
                model, dr_params, data, state, actions
            )
            info = dict(info) if info is not None else {}

            step_count = step_count + 1
            timed_out = step_count >= max_steps

            bad = jnp.any(jnp.isnan(new_data.qpos), axis=1) | jnp.any(jnp.isnan(obs), axis=1)
            rewards = jnp.where(bad, 0.0, jnp.where(jnp.isnan(rewards), 0.0, rewards))
            obs = jnp.where(bad[:, None], 0.0, jnp.nan_to_num(obs, nan=0.0))
            info["metrics/nan_rate"] = bad.astype(jnp.float32)

            truncated_only = timed_out & ~dones
            dones = dones | timed_out | bad

            if noise_std > 0.0:
                noise_key, subkey = jax.random.split(noise_key)
                obs = obs + jax.random.normal(subkey, obs.shape) * noise_std

            is_success = info.pop("is_success", None)
            means = {
                k: jnp.nan_to_num(jnp.mean(v), nan=0.0, posinf=0.0, neginf=0.0)
                for k, v in info.items()
            }
            return (
                new_data,
                new_state,
                obs,
                rewards,
                dones,
                truncated_only,
                step_count,
                noise_key,
                is_success,
                means,
            )

        donate = (2, 3) if jax.default_backend() in ("gpu", "cuda") else ()
        return jax.jit(_fused, donate_argnums=donate)

    def _build_fused_reset(self) -> Any:
        dr_enabled = self._dr_config.enabled
        noise_std = self._obs_noise_std
        vreset = jax.vmap(self._reset_single, in_axes=(None, 0, 0))
        vobs = jax.vmap(self._get_obs_single, in_axes=(None, 0, 0))

        def _fused(
            model: Any,
            dr_params: DRParams | None,
            dr_key: jax.Array,
            data: Any,
            state: Any,
            keys: jax.Array,
            step_count: jax.Array,
            noise_key: jax.Array,
            needs_reset: jax.Array,
        ) -> Any:
            split_keys = jax.vmap(jax.random.split)(keys)
            new_keys = jnp.where(needs_reset[:, None], split_keys[:, 0], keys)

            reset_data, reset_state = vreset(model, data, new_keys)
            new_data = _merge(needs_reset, reset_data, data)
            new_state = _merge(needs_reset, reset_state, state)

            if dr_enabled:
                dr_key, dr_subkey = jax.random.split(dr_key)
                reset_dr = self._sample_dr_params_batch(dr_subkey)
                dr_params = _merge(needs_reset, reset_dr, dr_params)

            step_count = jnp.where(needs_reset, 0, step_count)

            reset_obs = vobs(model, new_data, new_state)
            if noise_std > 0.0:
                noise_key, subkey = jax.random.split(noise_key)
                reset_obs = reset_obs + jax.random.normal(subkey, reset_obs.shape) * noise_std

            return (
                new_data,
                new_state,
                new_keys,
                dr_params,
                dr_key,
                step_count,
                noise_key,
                reset_obs,
            )

        donate = (3, 4) if jax.default_backend() in ("gpu", "cuda") else ()
        return jax.jit(_fused, donate_argnums=donate)

    def _build_model(self) -> mujoco.MjModel:
        raise NotImplementedError

    def _reset_single(self, mjx_model: Any, mjx_data: Any, key: jax.Array) -> Any:
        raise NotImplementedError

    def _step_single(self, mjx_model: Any, mjx_data: Any, env_state: Any, action: jax.Array) -> Any:
        raise NotImplementedError

    def _get_obs_single(self, mjx_model: Any, mjx_data: Any, env_state: Any) -> jax.Array:
        raise NotImplementedError

    def _obs_size(self) -> int:
        raise NotImplementedError

    def _action_size(self) -> int:
        raise NotImplementedError

    @property
    def _max_episode_steps(self) -> int:
        raise NotImplementedError

    def _sample_dr_params_single(self, key: jax.Array) -> DRParams:
        nbody = int(self._mj_model.nbody)
        ngeom = int(self._mj_model.ngeom)
        nu = int(self._mj_model.nu)
        if not self._dr_config.enabled:
            return DRParams(
                mass_mult=jnp.ones(nbody),
                friction_mult=jnp.ones(ngeom),
                gain_mult=jnp.ones(nu),
            )

        k_m, k_f, k_g = jax.random.split(key, 3)
        m_lo, m_hi = self._dr_config.mass_range
        f_lo, f_hi = self._dr_config.friction_range
        g_lo, g_hi = self._dr_config.actuator_gain_range
        return DRParams(
            mass_mult=jax.random.uniform(k_m, shape=(nbody,), minval=m_lo, maxval=m_hi),
            friction_mult=jax.random.uniform(k_f, shape=(ngeom,), minval=f_lo, maxval=f_hi),
            gain_mult=jax.random.uniform(k_g, shape=(nu,), minval=g_lo, maxval=g_hi),
        )

    def _sample_dr_params_batch(self, key: jax.Array) -> DRParams:
        keys = jax.random.split(key, self._num_envs)
        return jax.vmap(self._sample_dr_params_single)(keys)

    def _noisy_obs(self, obs: jax.Array) -> np.ndarray:
        if self._obs_noise_std <= 0.0:
            return np.array(obs)
        self._obs_noise_key, subkey = jax.random.split(self._obs_noise_key)
        noise = jax.random.normal(subkey, obs.shape) * self._obs_noise_std
        return np.array(obs + noise)

    def reset(self) -> VecEnvObs:
        base_data = mjx.make_data(self._mjx_model)
        batch_data = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self._num_envs, *x.shape)),
            base_data,
        )

        self._env_keys = jax.device_put(
            jax.random.split(
                jax.random.fold_in(self._master_key, self._full_reset_count), self._num_envs
            )
        )
        self._full_reset_count += 1

        if self._dr_config.enabled:
            self._dr_key, dr_subkey = jax.random.split(self._dr_key)
            self._dr_key = jax.device_put(self._dr_key)
            self._dr_params_batch = jax.device_put(self._sample_dr_params_batch(dr_subkey))
        else:
            self._dr_params_batch = None

        batch_data, env_state = self._batched_reset(self._mjx_model, batch_data, self._env_keys)
        self._mjx_data_batch = batch_data
        self._env_state_batch = env_state
        self._step_count = jax.device_put(jnp.zeros(self._num_envs, dtype=jnp.int32))

        obs = self._batched_get_obs(self._mjx_model, batch_data, env_state)
        return self._noisy_obs(obs)

    def step_async(self, actions: np.ndarray) -> None:
        actions_jax = jnp.asarray(actions, dtype=jnp.float32)

        (
            new_data,
            new_state,
            obs,
            rewards,
            dones,
            truncated_only,
            step_count,
            noise_key,
            is_success,
            means,
        ) = self._fused_step(
            self._mjx_model,
            self._dr_params_batch,
            self._mjx_data_batch,
            self._env_state_batch,
            actions_jax,
            self._step_count,
            self._obs_noise_key,
        )
        self._obs_noise_key = noise_key

        obs_np, rewards_np, dones_np, truncated_np, is_success_np, means_np = jax.device_get(
            (obs, rewards, dones, truncated_only, is_success, means)
        )

        if dones_np.any():
            (
                new_data,
                new_state,
                self._env_keys,
                self._dr_params_batch,
                self._dr_key,
                step_count,
                self._obs_noise_key,
                reset_obs,
            ) = self._fused_reset(
                self._mjx_model,
                self._dr_params_batch,
                self._dr_key,
                new_data,
                new_state,
                self._env_keys,
                step_count,
                self._obs_noise_key,
                dones,
            )
            reset_obs_np = np.asarray(reset_obs)
            obs_np = np.array(obs_np)
        else:
            reset_obs_np = None

        self._mjx_data_batch = new_data
        self._env_state_batch = new_state
        self._step_count = step_count

        rewards_np = np.asarray(rewards_np, dtype=np.float64)
        agg_info = {k: float(means_np[k]) for k in sorted(means_np)}

        infos: list[dict[str, Any]] = []
        for i in range(self._num_envs):
            info: dict[str, Any] = {}
            if dones_np[i]:
                if is_success_np is not None:
                    info["is_success"] = float(is_success_np[i])
                info["terminal_observation"] = obs_np[i].copy()
                if truncated_np[i]:
                    info["TimeLimit.truncated"] = True
                if reset_obs_np is not None:
                    obs_np[i] = reset_obs_np[i]

            infos.append(info)

        infos[0].update(agg_info)

        self._pending_obs = obs_np
        self._pending_rewards = rewards_np
        self._pending_dones = dones_np
        self._pending_infos = infos

    def step_wait(self) -> VecEnvStepReturn:
        assert self._pending_obs is not None
        obs = self._pending_obs
        rewards = self._pending_rewards
        dones = self._pending_dones
        infos = self._pending_infos

        self._pending_obs = None
        self._pending_rewards = None
        self._pending_dones = None
        self._pending_infos = None

        return obs, rewards, dones, infos  # type: ignore[return-value]

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> list[bool]:  # type: ignore[override]
        del wrapper_class, indices
        return [False] * self._num_envs

    def env_method(  # type: ignore[override]
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if indices is None:
            indices = range(self._num_envs)

        if hasattr(self, method_name):
            fn = getattr(self, method_name)
            return [fn(*method_args, **method_kwargs)] * len(list(indices))

        return [None] * len(list(indices))

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:  # type: ignore[override]
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            n = self._num_envs if indices is None else len(indices)
            return [val] * n

        return [None] * (self._num_envs if indices is None else len(indices))

    def set_attr(  # type: ignore[override]
        self, attr_name: str, value: Any, indices: Any = None
    ) -> None:
        setattr(self, attr_name, value)

    def seed(self, seed: int | None = None) -> None:  # type: ignore[override]
        if seed is not None:
            self._master_key = jax.random.PRNGKey(seed)
