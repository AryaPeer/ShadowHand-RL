import math

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from shadow_hand.config import PegRewardConfig, RewardConfig
from shadow_hand.rewards.grasp import (
    grasp_reward,
    init_grasp_reward_state,
)
from shadow_hand.rewards.peg import (
    init_peg_reward_state,
    peg_reward,
)


def _grasp_kwargs(cfg: RewardConfig, table_height: float) -> dict:
    return {
        "finger_positions": jnp.zeros((5, 3)),
        "object_position": jnp.array([0.0, 0.0, 0.5]),
        "object_linear_velocity": jnp.zeros(3),
        "finger_contact_mask": jnp.array([True, True, True, False, False]),
        "actions": jnp.zeros(23),
        "table_height": table_height,
        "cfg": cfg,
    }


class TestGraspJax:
    def test_jit_compiles(self):
        cfg = RewardConfig()

        @jax.jit
        def _run(
            state,
            finger_positions,
            object_position,
            object_linear_velocity,
            finger_contact_mask,
            actions,
        ):
            return grasp_reward(
                state=state,
                finger_positions=finger_positions,
                object_position=object_position,
                object_linear_velocity=object_linear_velocity,
                finger_contact_mask=finger_contact_mask,
                actions=actions,
                table_height=0.4,
                cfg=cfg,
            )

        state = init_grasp_reward_state(0.4, 0.4)
        total, _, info = _run(
            state,
            jnp.zeros((5, 3)),
            jnp.array([0.0, 0.0, 0.5]),
            jnp.zeros(3),
            jnp.array([True, True, True, False, False]),
            jnp.zeros(23),
        )
        assert np.isfinite(float(total))
        assert "reward/total" in info

    def test_vmap_over_batch(self):
        cfg = RewardConfig()
        B = 8

        def _one(state, obj_z):
            kw = _grasp_kwargs(cfg, 0.4)
            kw["object_position"] = jnp.array([0.0, 0.0, obj_z])
            total, _, _ = grasp_reward(state=state, **kw)
            return total

        batched = jax.vmap(_one, in_axes=(0, 0))
        states = jax.vmap(lambda z: init_grasp_reward_state(z, 0.4))(jnp.linspace(0.4, 0.6, B))
        totals = batched(states, jnp.linspace(0.4, 0.6, B))
        assert totals.shape == (B,)
        assert bool(jnp.all(jnp.isfinite(totals)))

    def test_lifted_latch_advances(self):
        cfg = RewardConfig()
        kw = _grasp_kwargs(cfg, 0.4)
        kw["object_position"] = jnp.array([0.0, 0.0, 0.55])
        state = init_grasp_reward_state(0.4, 0.4)
        _, new_state, _ = grasp_reward(state=state, **kw)
        assert bool(new_state.was_lifted)

        kw["object_position"] = jnp.array([0.0, 0.0, 0.405])
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, final_state, info = grasp_reward(state=new_state, **kw)

        assert float(info["reward/drop"]) < 0.0

    def test_success_annuity_pays_while_held(self):
        cfg = RewardConfig()
        held = _grasp_kwargs(cfg, 0.4)
        held["object_position"] = jnp.array([0.0, 0.0, 0.55])
        dropped = _grasp_kwargs(cfg, 0.4)
        dropped["object_position"] = jnp.array([0.0, 0.0, 0.436])
        dropped["finger_contact_mask"] = jnp.array([False] * 5)

        state = init_grasp_reward_state(0.436, 0.4)
        bonuses = []
        for _ in range(cfg.success_hold_steps + 5):
            _, state, info = grasp_reward(state=state, **held)
            bonuses.append(float(info["reward/success"]))

        assert all(b == 0.0 for b in bonuses[: cfg.success_hold_steps - 1])
        assert all(b == cfg.success_bonus_per_step for b in bonuses[cfg.success_hold_steps - 1 :])

        _, state, info = grasp_reward(state=state, **dropped)
        assert float(info["reward/success"]) == 0.0
        assert float(info["reward/drop"]) < 0.0

        for _ in range(2):
            _, state, info = grasp_reward(state=state, **dropped)
            assert float(info["reward/success"]) == 0.0

        rehold_bonuses = []
        for _ in range(cfg.success_hold_steps + 5):
            _, state, info = grasp_reward(state=state, **held)
            rehold_bonuses.append(float(info["reward/success"]))

        assert all(b == 0.0 for b in rehold_bonuses[: cfg.success_hold_steps - 1])
        assert all(
            b == cfg.success_bonus_per_step for b in rehold_bonuses[cfg.success_hold_steps - 1 :]
        )
        assert float(info["is_success"]) == 1.0


class TestPegJax:
    def _kw(self) -> dict:
        cfg = PegRewardConfig()
        peg_length = 0.06
        return {
            "stage": jnp.asarray(0),
            "finger_positions": jnp.zeros((5, 3)),
            "peg_position": jnp.array([0.0, 0.0, 0.9]),
            "peg_axis": jnp.array([0.0, 0.0, 1.0]),
            "peg_linear_velocity": jnp.zeros(3),
            "hole_position": jnp.array([0.0, 0.0, 0.88]),
            "hole_axis": jnp.array([0.0, 0.0, 1.0]),
            "insertion_depth": jnp.asarray(0.0),
            "contact_force_magnitude": jnp.asarray(0.0),
            "finger_contact_mask": jnp.array([True, True, True, False, False]),
            "peg_height": jnp.asarray(0.9),
            "actions": jnp.zeros(23),
            "peg_length": peg_length,
            "table_height": 0.82,
            "cfg": cfg,
        }

    def test_jit_compiles(self):
        kw = self._kw()
        state = init_peg_reward_state(0.85)

        @jax.jit
        def _run(state):
            return peg_reward(state=state, **kw)

        total, _, info = _run(state)
        assert np.isfinite(float(total))
        assert "reward/depth" in info
        assert "reward/place_release" in info

    def test_success_annuity_fires_after_hold(self):
        cfg = PegRewardConfig()
        kw = self._kw()
        kw["insertion_depth"] = jnp.asarray(0.05)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        state = init_peg_reward_state(0.85)

        bonuses = []
        for _ in range(cfg.peg_hold_steps + 3):
            _, state, info = peg_reward(state=state, **kw)
            bonuses.append(float(info["reward/success"]))

        assert bonuses[0] == 0.0
        assert all(b == 0.0 for b in bonuses[: cfg.peg_hold_steps - 1])
        assert all(b == cfg.success_bonus_per_step for b in bonuses[cfg.peg_hold_steps - 1 :])

    def test_success_requires_release(self):
        kw = self._kw()
        kw["insertion_depth"] = jnp.asarray(0.05)
        kw["finger_contact_mask"] = jnp.array([True, True, True, True, True])
        state = init_peg_reward_state(0.85)
        for _ in range(30):
            _, state, info = peg_reward(state=state, **kw)
        assert float(info["metrics/insertion_hold_steps"]) == 0.0
        assert float(info["reward/success"]) == 0.0

    def test_place_release_requires_release(self):
        kw = self._kw()
        kw["insertion_depth"] = jnp.asarray(0.02)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, _, released = peg_reward(state=init_peg_reward_state(0.85), **kw)

        kw["finger_contact_mask"] = jnp.array([True, True, True, True, True])
        _, _, gripped = peg_reward(state=init_peg_reward_state(0.85), **kw)

        assert float(released["reward/place_release"]) > 0.5
        assert float(gripped["reward/place_release"]) == 0.0

    def test_holding_requires_lift_and_grip(self):
        def holding_at(peg_z, mask):
            kw = self._kw()
            kw["peg_position"] = jnp.array([0.0, 0.0, peg_z])
            kw["peg_height"] = jnp.asarray(peg_z)
            kw["finger_contact_mask"] = jnp.asarray(mask)
            _, _, info = peg_reward(state=init_peg_reward_state(0.85), **kw)
            return float(info["reward/holding"])

        gripped = [True, True, True, False, False]
        lifted_gripped = holding_at(0.92, gripped)
        not_lifted = holding_at(0.85, gripped)
        lifted_ungripped = holding_at(0.92, [False] * 5)
        assert lifted_gripped > not_lifted
        assert lifted_gripped > lifted_ungripped

    def test_place_monotone_toward_engaged_pose(self):
        cfg = PegRewardConfig()
        peg_length = 0.06
        hole_z = 0.88

        def place_at(peg_xy, peg_z):
            kw = self._kw()
            kw["peg_position"] = jnp.array([peg_xy[0], peg_xy[1], peg_z])
            kw["peg_height"] = jnp.asarray(peg_z)
            _, _, info = peg_reward(state=init_peg_reward_state(0.85), **kw)
            return float(info["reward/place"])

        engaged_z = hole_z + cfg.release_height + peg_length / 2.0
        p_table = place_at((0.05, 0.0), 0.85)
        p_lifted = place_at((0.05, 0.0), 0.90)
        p_engaged = place_at((0.0, 0.0), engaged_z)
        assert p_table < p_engaged
        assert p_lifted < p_engaged
        assert p_engaged > 0.9

    def test_place_pays_nothing_ungripped_outside_bore(self):
        kw = self._kw()
        kw["peg_position"] = jnp.array([0.05, 0.0, 0.85])
        kw["peg_height"] = jnp.asarray(0.85)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, _, info = peg_reward(state=init_peg_reward_state(0.85), **kw)
        assert float(info["reward/place"]) < 0.02, (
            "an ungripped peg parked outside the bore must not earn place"
        )

    def test_drop_penalty_not_fired_when_inserted(self):
        kw = self._kw()
        state = init_peg_reward_state(0.85)
        state = state._replace(was_lifted=jnp.array(True))
        kw["peg_position"] = jnp.array([0.0, 0.0, 0.855])
        kw["peg_height"] = jnp.asarray(0.855)
        kw["insertion_depth"] = jnp.asarray(0.045)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, _, info = peg_reward(state=state, **kw)
        assert float(info["reward/drop"]) == 0.0

    def test_depth_requires_release(self):
        kw = self._kw()
        kw["insertion_depth"] = jnp.asarray(0.045)
        kw["finger_contact_mask"] = jnp.array([True, True, True, True, True])
        _, _, gripped = peg_reward(state=init_peg_reward_state(0.85), **kw)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, _, released = peg_reward(state=init_peg_reward_state(0.85), **kw)
        assert float(gripped["reward/depth"]) == 0.0
        assert float(released["reward/depth"]) > 0.0

    def test_drop_penalty_exempt_near_hole(self):
        kw = self._kw()
        state = init_peg_reward_state(0.85)._replace(was_lifted=jnp.array(True))
        kw["peg_height"] = jnp.asarray(0.851)
        kw["insertion_depth"] = jnp.asarray(0.0)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        kw["peg_position"] = jnp.array([0.0, 0.0, 0.851])
        _, _, near = peg_reward(state=state, **kw)
        assert float(near["reward/drop"]) == 0.0
        kw["peg_position"] = jnp.array([0.15, 0.0, 0.851])
        _, _, far = peg_reward(state=state, **kw)
        assert float(far["reward/drop"]) < 0.0

    def test_premature_release_penalizes_far_drop(self):
        kw = self._kw()
        kw["peg_position"] = jnp.array([0.15, 0.0, 0.9])
        kw["peg_height"] = jnp.asarray(0.9)
        kw["insertion_depth"] = jnp.asarray(0.0)

        kw["finger_contact_mask"] = jnp.array([True, True, True, True, True])
        _, gripped_state, _ = peg_reward(state=init_peg_reward_state(0.85), **kw)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, _, far = peg_reward(state=gripped_state, **kw)
        assert float(far["reward/premature_release"]) < 0.0

        _, _, never = peg_reward(state=init_peg_reward_state(0.85), **kw)
        assert float(never["reward/premature_release"]) == 0.0

        kw["peg_position"] = jnp.array([0.0, 0.0, 0.9])
        kw["finger_contact_mask"] = jnp.array([True, True, True, True, True])
        _, near_state, _ = peg_reward(state=init_peg_reward_state(0.85), **kw)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, _, near = peg_reward(state=near_state, **kw)
        assert float(near["reward/premature_release"]) == 0.0

    def test_depth_reward_clamped(self):
        kw = self._kw()
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        kw["insertion_depth"] = jnp.asarray(0.5)
        state = init_peg_reward_state(0.85)
        _, _, info = peg_reward(state=state, **kw)
        assert float(info["reward/depth"]) <= 10.0 + 1e-6
        assert float(info["reward/depth"]) > 0.0

        kw["insertion_depth"] = jnp.asarray(-0.1)
        _, _, info_neg = peg_reward(state=state, **kw)
        assert float(info_neg["reward/depth"]) == 0.0

    def test_force_penalty_bounded(self):
        kw = self._kw()
        kw["contact_force_magnitude"] = jnp.asarray(1_000_000.0)
        _, _, info = peg_reward(state=init_peg_reward_state(0.85), **kw)
        assert float(info["reward/force_penalty"]) >= -4.0 - 1e-6, (
            "force penalty must be bounded so a jam cannot dominate the reward"
        )

    def test_force_penalty_targets_jamming_not_insertion(self):
        kw = self._kw()
        kw["contact_force_magnitude"] = jnp.asarray(300.0)
        _, _, low = peg_reward(state=init_peg_reward_state(0.85), **kw)
        assert float(low["reward/force_penalty"]) == 0.0, (
            "working-insertion contact (below threshold) must not be penalized"
        )
        kw["contact_force_magnitude"] = jnp.asarray(3000.0)
        _, _, jam = peg_reward(state=init_peg_reward_state(0.85), **kw)
        assert float(jam["reward/force_penalty"]) < -2.0, (
            "jam-level contact force must be strongly penalized"
        )

    def _deep_released(self, tilt: float) -> dict:
        kw = self._kw()
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        kw["insertion_depth"] = jnp.asarray(0.05)
        kw["peg_axis"] = jnp.array([math.sin(tilt), 0.0, math.cos(tilt)])
        return kw

    def _hold_to_success(self, kw: dict) -> dict:
        state = init_peg_reward_state(0.85)
        info: dict = {}
        for _ in range(PegRewardConfig().peg_hold_steps + 3):
            _, state, info = peg_reward(state=state, **kw)
        return info

    def test_crooked_insertion_cannot_earn_success(self):
        straight = self._hold_to_success(self._deep_released(0.0))
        assert float(straight["reward/success"]) > 0.0, (
            "an aligned deep insertion must still reach success"
        )
        crooked = self._hold_to_success(self._deep_released(math.radians(40.0)))
        assert float(crooked["reward/success"]) == 0.0, (
            "a 40deg-tilted peg must not count as success (axis_align 0.77 < 0.85)"
        )
        assert float(crooked["metrics/insertion_hold_steps"]) == 0.0, (
            "the hold counter must not accumulate while misaligned"
        )

    def test_moving_peg_cannot_bank_success(self):
        settled = self._hold_to_success(self._deep_released(0.0))
        assert float(settled["reward/success"]) > 0.0

        kw = self._deep_released(0.0)
        kw["peg_linear_velocity"] = jnp.array([0.0, 0.0, -0.5])
        moving = self._hold_to_success(kw)
        assert float(moving["reward/success"]) == 0.0, (
            "a peg still moving at 0.5 m/s must not count as a completed insertion"
        )
        assert float(moving["metrics/insertion_hold_steps"]) == 0.0

    def test_alignment_is_rewarded_after_release(self):
        straight = self._deep_released(0.0)
        crooked = self._deep_released(math.radians(30.0))
        _, _, s_info = peg_reward(state=init_peg_reward_state(0.85), **straight)
        _, _, c_info = peg_reward(state=init_peg_reward_state(0.85), **crooked)
        assert float(s_info["reward/align"]) > float(c_info["reward/align"]) > 0.0, (
            "align must survive release so straightness is rewarded during insertion"
        )
