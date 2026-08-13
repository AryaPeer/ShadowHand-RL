from shadow_hand.config import (
    AdaptiveCurriculumConfig,
    MjxGraspTrainConfig,
    MjxPegTrainConfig,
    PegRewardConfig,
    PegRewardWeights,
    PegSceneConfig,
    RewardConfig,
    RewardWeights,
    SceneConfig,
)


class TestConfigDefaults:
    def test_scene_config(self):
        c = SceneConfig()
        assert c.mount_x == -0.10
        assert c.mount_y == 0.0
        assert c.mount_height == 0.78
        assert c.sim_timestep == 0.005
        assert c.frame_skip == 8
        assert c.solver_iterations == 8
        assert c.ls_iterations == 8

    def test_reward_weights(self):
        w = RewardWeights()
        assert w.reaching == 0.5
        assert w.grasping == 2.5
        assert w.lifting == 6.0
        assert w.holding == 6.0

    def test_reward_config(self):
        c = RewardConfig()
        assert isinstance(c.weights, RewardWeights)
        assert c.lift_target == 0.10
        assert c.hold_height_smoothness_k == 50.0
        assert c.hold_velocity_smoothness_k == 100.0
        assert c.no_contact_idle_penalty == -0.08
        assert c.success_bonus_per_step == 5.0
        assert c.drop_arm_height == 0.04
        assert c.fingertip_weights[4] == max(c.fingertip_weights)

    def test_peg_scene_config(self):
        c = PegSceneConfig()
        assert c.mount_x == -0.10
        assert c.mount_y == 0.0
        assert c.mount_height == 0.82
        assert c.action_smoothing_alpha == 0.2
        assert c.spawn_min_radius == 0.020
        assert c.spawn_max_radius == 0.028
        assert len(c.pick_center) == 2
        sep = abs(c.hole_offset[0] - c.pick_center[0])
        assert sep - c.spawn_max_radius >= 0.08, (
            "every spawn must clear the socket by the measured hand-reach margin"
        )
        assert c.clearance == 0.003
        assert c.hole_depth == 0.050
        assert c.hole_top_above_table == 0.056
        assert c.hole_top_above_table - c.hole_depth > 0.0025, (
            "pedestal degenerates unless hole_top clears hole_depth by > 2.5mm"
        )
        bore_ratio = c.hole_depth / (2 * (c.peg_radius + c.clearance))
        assert bore_ratio > 1.0, (
            f"a tube must be deeper than it is wide; depth:width = {bore_ratio:.2f}"
        )
        assert len(c.hole_offset) == 2
        assert c.peg_radius == 0.015
        assert c.peg_half_length == 0.025
        assert c.peg_mass == 0.02
        assert not hasattr(c, "peg_com_offset"), (
            "peg_com_offset (bottom-weight) removed — its only purpose was free-cylinder "
            "tip resistance for the from-table pick, which is gone (100% gripped now)"
        )
        aspect = (2 * c.peg_half_length) / (2 * c.peg_radius)
        assert aspect <= 2.0, (
            f"peg aspect {aspect:.2f} — a slender peg cannot be held by a formed grip "
            f"at table height (measured: 16x60mm holds 0/40 steps, 30x50mm holds 40/40)"
        )
        assert c.solver_iterations == 8
        assert c.ls_iterations == 8

    def test_peg_reward_config(self):
        c = PegRewardConfig()
        assert c.force_threshold == 600.0
        assert c.weights.opposition == 1.0
        assert c.weights.axis_in_grip == 1.5
        assert c.weights.place == 8.0
        assert c.weights.holding == 2.0
        assert c.weights.place_release == 22.0
        assert c.release_height == -0.015
        assert c.lateral_gate_k == 5.0
        assert c.peg_hold_steps == 10
        assert c.success_threshold == 0.6
        assert c.carry_clear_height == 0.02

    def test_mjx_peg_adaptive_curriculum(self):
        c = MjxPegTrainConfig()
        assert isinstance(c.scene_config, PegSceneConfig)
        assert isinstance(c.reward_config, PegRewardConfig)
        assert not hasattr(c, "curriculum_stages")
        ac = c.adaptive_curriculum
        assert isinstance(ac, AdaptiveCurriculumConfig)
        assert ac.retreat_success_frac < ac.advance_success_frac
        levels = ac.carry_floor_levels
        assert len(levels) >= 2
        assert list(levels) == sorted(levels)
        assert all(0.0 <= x < 1.0 for x in levels)
        assert levels[0] == 0.0, "SBC must expose the full spawn range from step 0"
        assert ac.check_interval_rollouts >= 1
        assert ac.success_window_rollouts >= 1

        for name in (
            "plateau_stop_timestep",
            "plateau_carry_floor_min",
            "p_from_table",
            "p_from_table_levels",
        ):
            assert not hasattr(ac, name), (
                f"{name} should be removed — the task is now 100% gripped-start "
                f"insertion (IndustReal Insert policy); the from-table pick ladder is gone"
            )

    def test_peg_entropy_bonus_matches_solved_tasks(self):
        assert MjxPegTrainConfig().ent_coef == MjxGraspTrainConfig().ent_coef == 0.0

    def test_mjx_log_std_clamp_defaults(self):
        for cls in (MjxGraspTrainConfig, MjxPegTrainConfig):
            c = cls()
            assert c.log_std_min < c.log_std_max
            assert c.log_std_max == 0.0
            margin = 0.05
            assert c.log_std_min + margin < c.log_std_init < c.log_std_max - margin

    def test_all_configs_instantiate(self):

        configs = [
            SceneConfig,
            RewardWeights,
            RewardConfig,
            MjxGraspTrainConfig,
            PegSceneConfig,
            PegRewardWeights,
            PegRewardConfig,
            MjxPegTrainConfig,
        ]

        for cls in configs:
            obj = cls()
            assert obj is not None

    def test_removed_fields_stay_removed(self):
        w = RewardWeights()
        for name in ("action", "upward", "opposition"):
            assert not hasattr(w, name), f"RewardWeights.{name} should be removed"

        c = RewardConfig()
        for name in ("hold_bonus", "success_bonus"):
            assert not hasattr(c, name), f"RewardConfig.{name} should be removed"

        pw = PegRewardWeights()
        for name in (
            "upward",
            "action_magnitude",
            "insertion_drive",
            "complete",
            "idle_stage0",
            "idle_stage1",
        ):
            assert not hasattr(pw, name), f"PegRewardWeights.{name} should be removed"

        pc = PegRewardConfig()
        for name in (
            "min_contacts_for_align",
            "complete_bonus",
            "idle_stage0_penalty",
            "idle_stage1_penalty",
            "idle_stage1_min_contacts",
            "lift_step_threshold",
            "idle_stage_cutoff",
        ):
            assert not hasattr(pc, name), f"PegRewardConfig.{name} should be removed"
