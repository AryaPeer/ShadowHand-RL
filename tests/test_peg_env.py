import numpy as np
import pytest

pytest.importorskip("mujoco.mjx")
pytest.importorskip("jax")

from dexterous_hand.config import MjxPegTrainConfig, PegSceneConfig
from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv


@pytest.mark.slow
class TestPegMjxSmoke:
    def test_reset_and_step(self):
        env = ShadowHandPegMjxEnv(num_envs=4, seed=0, max_episode_steps=50)
        try:
            obs = env.reset()
            assert obs.shape == (4, env._obs_size())
            assert np.all(np.isfinite(obs))

            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            for _ in range(5):
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                assert obs.shape == (4, env._obs_size())
                assert rewards.shape == (4,)
                assert dones.shape == (4,)
                assert len(infos) == 4
                assert np.all(np.isfinite(obs))
                assert np.all(np.isfinite(rewards))
        finally:
            env.close()

    def test_curriculum_clearance_change_rebuilds_and_steps(self):
        env = ShadowHandPegMjxEnv(num_envs=2, seed=0, max_episode_steps=2)
        try:
            env.reset()
            env.set_curriculum_params(clearance=0.003, carry_floor=0.5)
            assert abs(float(env.scene_config.clearance) - 0.003) < 1e-12
            actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
            for step in range(1, 5):
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                assert np.all(np.isfinite(obs))
                assert np.all(np.isfinite(rewards))
                assert bool(dones.all()) == (step % 2 == 0)
        finally:
            env.close()

    def test_auto_reset_cycles_episodes(self):
        env = ShadowHandPegMjxEnv(num_envs=4, seed=0, max_episode_steps=3)
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

    def test_pregrasped_lift_reference_clamped_to_table(self):
        env = ShadowHandPegMjxEnv(num_envs=8, seed=0, max_episode_steps=50)
        try:
            env.set_curriculum_params(
                clearance=float(env.scene_config.clearance),
                carry_floor=0.0,
            )
            env.reset()
            cfg = env.scene_config
            table_spawn = cfg.table_height + cfg.peg_half_length + 0.001
            reward_init_h = np.asarray(env._env_state_batch.reward_state.initial_peg_height)
            assert np.all(reward_init_h <= table_spawn + 1e-6)
            peg_z = np.asarray(env._mjx_data_batch.xpos[:, env._nm.peg_body_id, 2])
            assert peg_z.max() > table_spawn + 0.02, "no gripped spawn reached the bore"
        finally:
            env.close()

    def test_sbc_spawn_spans_bore_to_table_from_step_zero(self):
        env = ShadowHandPegMjxEnv(num_envs=8, seed=0, max_episode_steps=50, carry_floor=0.0)
        try:
            env.reset()
            cfg = env.scene_config
            peg = np.asarray(env._mjx_data_batch.xpos[:, env._nm.peg_body_id, :])
            table_spawn = cfg.table_height + cfg.peg_half_length + 0.001
            entrance = cfg.table_height + cfg.hole_top_above_table

            max_lateral = abs(cfg.pick_center[0]) + cfg.spawn_max_radius + 0.02
            assert np.all(np.linalg.norm(peg[:, :2], axis=1) < max_lateral), (
                "gripped spawn strayed laterally off the carry path"
            )
            assert np.all(peg[:, 2] > table_spawn - 0.02), (
                "a gripped peg dropped through the table — grip not holding"
            )
            assert peg[:, 2].max() > entrance - 0.02, "no spawn near the bore mouth"
            assert peg[:, 2].max() - peg[:, 2].min() > 0.03, (
                "spawn heights did not span the carry range at carry_floor=0"
            )

            actions = np.zeros((8, env.action_space.shape[0]), dtype=np.float32)
            for _ in range(5):
                env.step_async(actions)
                _obs, rewards, _dones, _infos = env.step_wait()
                assert np.all(np.isfinite(rewards))
                assert np.all(rewards > -20.0), (
                    f"spawn step reward {rewards.min()} — jam / unbounded force penalty"
                )
        finally:
            env.close()

    def test_from_config_seeds_first_rollout_curriculum(self):
        config = MjxPegTrainConfig(num_envs=2, max_episode_steps=10)
        env = ShadowHandPegMjxEnv.from_config(config)
        try:
            ac = config.adaptive_curriculum
            assert float(env._carry_floor) == pytest.approx(ac.carry_floor_levels[0])
        finally:
            env.close()

    def test_set_curriculum_params_caches_reset_per_level(self):
        env = ShadowHandPegMjxEnv(num_envs=2, seed=0, max_episode_steps=5)
        try:
            cl = float(env.scene_config.clearance)
            env.set_curriculum_params(clearance=cl, carry_floor=0.40)
            r1, f1 = env._batched_reset, env._fused_reset
            env.set_curriculum_params(clearance=cl, carry_floor=0.70)
            assert env._batched_reset is not r1, "a new level must build a fresh reset"
            env.set_curriculum_params(clearance=cl, carry_floor=0.40)
            assert env._batched_reset is r1 and env._fused_reset is f1, (
                "a repeated level must reuse the cached reset (the recompile fix)"
            )
        finally:
            env.close()

    def test_spawn_never_interpenetrates_the_socket(self):
        env = ShadowHandPegMjxEnv(num_envs=48, seed=0, max_episode_steps=50, carry_floor=0.0)
        try:
            env.reset()
            sd = np.asarray(env._mjx_data_batch.sensordata)
            force = sd[:, np.asarray(env._wall_force_adr)].sum(axis=1)
            pz = np.asarray(env._mjx_data_batch.xpos[:, env._nm.peg_body_id, 2])
            worst = int(np.argmax(force))
            assert force.max() < 500.0, (
                f"spawn drives {force.max():.0f}N into the socket walls "
                f"(peg_z={pz[worst]:.4f}) — the carry path clips the socket"
            )
        finally:
            env.close()

    def test_carry_floor_removes_easy_spawns(self):
        cfg = PegSceneConfig()
        engaged_z = cfg.table_height + cfg.hole_top_above_table + 0.01 + cfg.peg_half_length
        table_z = cfg.table_height + cfg.peg_half_length + 0.001
        span = engaged_z - table_z

        bore_r = cfg.peg_radius + cfg.clearance
        hx, hy = cfg.hole_offset

        def probe(carry_floor):
            env = ShadowHandPegMjxEnv(
                num_envs=48,
                seed=0,
                max_episode_steps=50,
                carry_floor=carry_floor,
                scene_config=PegSceneConfig(),
            )

            try:
                env.reset()
                xp = np.asarray(env._mjx_data_batch.xpos[:, env._nm.peg_body_id, :])
            finally:
                env.close()

            return xp[:, 2], np.hypot(xp[:, 0] - hx, xp[:, 1] - hy)

        full, full_lat = probe(0.0)
        raised, raised_lat = probe(0.9)
        assert full.max() - full.min() > 0.5 * span, (
            f"carry_floor=0 must span most of the bore->table range; covered "
            f"{(full.max() - full.min()) / span:.0%} of {span * 1000:.0f}mm"
        )
        assert raised.max() - raised.min() < 0.25 * span, (
            f"carry_floor=0.9 must confine spawns to the table end; band is "
            f"{(raised.max() - raised.min()) / span:.0%} of the range"
        )
        assert raised.max() < full.mean(), (
            "the raised band must sit below the middle of the full range"
        )
        assert full_lat.min() < bore_r, (
            f"carry_floor=0 must put SOME spawn over the bore, else the easy end of the "
            f"SBC ladder does not exist; closest was {full_lat.min() * 1000:.1f}mm vs "
            f"bore half-width {bore_r * 1000:.1f}mm. Height alone is not enough — the "
            f"lateral travel is {abs(hx - cfg.pick_center[0]) * 1000:.0f}mm."
        )
        assert full_lat.max() > 0.08, (
            f"carry_floor=0 must still reach the table end; furthest was "
            f"{full_lat.max() * 1000:.1f}mm"
        )
        assert raised_lat.min() > bore_r, (
            f"carry_floor=0.9 must delete every over-bore spawn; closest was "
            f"{raised_lat.min() * 1000:.1f}mm vs bore half-width {bore_r * 1000:.1f}mm"
        )
