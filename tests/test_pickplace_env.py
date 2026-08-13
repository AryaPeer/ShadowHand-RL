import numpy as np
import pytest

pytest.importorskip("mujoco.mjx")
pytest.importorskip("jax")

from shadow_hand.envs.pickplace_env import ShadowHandPickPlaceMjxEnv


@pytest.mark.slow
class TestPickPlaceMjxSmoke:
    def test_reset_and_step(self):
        env = ShadowHandPickPlaceMjxEnv(num_envs=4, seed=0, max_episode_steps=50)
        try:
            obs = env.reset()
            assert obs.shape == (4, 117)
            assert np.all(np.isfinite(obs))

            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            for _ in range(5):
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                assert obs.shape == (4, 117)
                assert rewards.shape == (4,)
                assert dones.shape == (4,)
                assert np.all(np.isfinite(obs))
                assert np.all(np.isfinite(rewards))
        finally:
            env.close()

    def test_goal_marker_varies_per_env(self):
        env = ShadowHandPickPlaceMjxEnv(num_envs=8, seed=0, max_episode_steps=50)
        try:
            env.reset()
            mocap = np.asarray(env._mjx_data_batch.mocap_pos)
            gid = env._nm.goal_mocap_id
            goals = mocap[:, gid, :2]
            assert goals.shape == (8, 2)
            assert np.ptp(goals[:, 1]) > 0.0
        finally:
            env.close()

    def test_winnable_cpu(self):
        from scripts.mjx_parity_check import (
            PICKPLACE_PLACE_BAR,
            PICKPLACE_Z_BAR,
            CpuEngine,
            run_pickplace,
        )

        r = run_pickplace(CpuEngine)
        assert r["place_dist"] <= PICKPLACE_PLACE_BAR
        assert r["obj_z_err"] <= PICKPLACE_Z_BAR
