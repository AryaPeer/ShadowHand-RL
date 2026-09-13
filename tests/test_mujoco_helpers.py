import pytest

from shadow_hand.scenes.grasp import build_grasp_scene
from shadow_hand.utils.mujoco_helpers import get_joint_qpos_qvel_range


@pytest.mark.slow
def test_get_joint_qpos_qvel_range_returns_contiguous_block():
    model, _, nm = build_grasp_scene()
    qpos_start, qpos_end, qvel_start, qvel_end = get_joint_qpos_qvel_range(model, nm.hand_joint_ids)
    assert qpos_start < qpos_end
    assert qvel_start < qvel_end
    assert qpos_end - qpos_start == nm.hand_qpos_end - nm.hand_qpos_start
    assert qvel_end - qvel_start == nm.hand_qvel_end - nm.hand_qvel_start
