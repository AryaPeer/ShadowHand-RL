import numpy as np
import pytest

pytest.importorskip("mujoco.mjx")
pytest.importorskip("jax")

from shadow_hand.envs.grasp_env import ShadowHandGraspMjxEnv


@pytest.mark.slow
class TestGraspMjxSmoke:
    def test_reset_and_step(self):
        env = ShadowHandGraspMjxEnv(num_envs=4, seed=0, max_episode_steps=50)
        try:
            obs = env.reset()
            assert obs.shape == (4, 108)
            assert np.all(np.isfinite(obs))

            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            for _ in range(5):
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                assert obs.shape == (4, 108)
                assert rewards.shape == (4,)
                assert dones.shape == (4,)
                assert len(infos) == 4
                assert np.all(np.isfinite(obs))
                assert np.all(np.isfinite(rewards))
        finally:
            env.close()

    def test_auto_reset_cycles_episodes(self):
        env = ShadowHandGraspMjxEnv(num_envs=4, seed=0, max_episode_steps=3)
        try:
            env.reset()
            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            for step in range(1, 9):
                env.step_async(actions)
                obs, _rewards, dones, infos = env.step_wait()
                expect_done = step % 3 == 0
                assert bool(dones.all()) == expect_done
                assert bool(dones.any()) == expect_done

                for i, info in enumerate(infos):
                    if expect_done:
                        assert info["TimeLimit.truncated"] is True
                        assert "is_success" in info
                        term = info["terminal_observation"]
                        assert term.shape == obs[i].shape
                        assert np.all(np.isfinite(term))
                        assert not np.allclose(term, obs[i])
                    else:
                        assert "terminal_observation" not in info
                        assert "is_success" not in info

                assert np.all(np.isfinite(obs))
        finally:
            env.close()
