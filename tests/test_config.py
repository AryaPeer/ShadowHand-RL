from shadow_hand.config import (
    AdaptiveCurriculumConfig,
    MjxGraspTrainConfig,
    MjxPegTrainConfig,
    PegRewardConfig,
    PegRewardWeights,
    PegSceneConfig,
    RewardConfig,
    RewardWeights,
)


class TestConfigDefaults:
    def test_reward_config(self):
        c = RewardConfig()
        assert c.fingertip_weights[4] == max(c.fingertip_weights)

    def test_peg_scene_config(self):
        c = PegSceneConfig()
        assert len(c.pick_center) == 2
        sep = abs(c.hole_offset[0] - c.pick_center[0])
        assert sep - c.spawn_max_radius >= 0.08, (
            "every spawn must clear the socket by the measured hand-reach margin"
        )
        assert c.hole_top_above_table - c.hole_depth > 0.0025, (
            "pedestal degenerates unless hole_top clears hole_depth by > 2.5mm"
        )
        bore_ratio = c.hole_depth / (2 * (c.peg_radius + c.clearance))
        assert bore_ratio > 1.0
        assert len(c.hole_offset) == 2
        assert not hasattr(c, "peg_com_offset"), (
            "peg_com_offset (bottom-weight) removed — its only purpose was free-cylinder "
            "tip resistance for the from-table pick, which is gone (100% gripped now)"
        )
        aspect = (2 * c.peg_half_length) / (2 * c.peg_radius)
        assert aspect <= 2.0

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
            assert not hasattr(ac, name)

    def test_peg_entropy_bonus_matches_solved_tasks(self):
        assert MjxPegTrainConfig().ent_coef == MjxGraspTrainConfig().ent_coef == 0.0

    def test_mjx_log_std_clamp_defaults(self):
        for cls in (MjxGraspTrainConfig, MjxPegTrainConfig):
            c = cls()
            assert c.log_std_min < c.log_std_max
            assert c.log_std_max == 0.0
            margin = 0.05
            assert c.log_std_min + margin < c.log_std_init < c.log_std_max - margin

    def test_removed_fields_stay_removed(self):
        w = RewardWeights()
        for name in ("action", "upward", "opposition"):
            assert not hasattr(w, name)

        c = RewardConfig()
        for name in ("hold_bonus", "success_bonus"):
            assert not hasattr(c, name)

        pw = PegRewardWeights()
        for name in (
            "upward",
            "action_magnitude",
            "insertion_drive",
            "complete",
            "idle_stage0",
            "idle_stage1",
        ):
            assert not hasattr(pw, name)

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
            assert not hasattr(pc, name)
