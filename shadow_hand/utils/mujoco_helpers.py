import mujoco

JOINT_QPOS_SIZE = {0: 7, 1: 4, 2: 1, 3: 1}
JOINT_DOF_SIZE = {0: 6, 1: 3, 2: 1, 3: 1}


def get_joint_qpos_qvel_range(
    model: mujoco.MjModel,
    joint_ids: list[int],
) -> tuple[int, int, int, int]:
    first, last = joint_ids[0], joint_ids[-1]
    last_type = int(model.jnt_type[last])

    qpos_start = int(model.jnt_qposadr[first])
    qpos_end = int(model.jnt_qposadr[last] + JOINT_QPOS_SIZE[last_type])
    qvel_start = int(model.jnt_dofadr[first])
    qvel_end = int(model.jnt_dofadr[last] + JOINT_DOF_SIZE[last_type])

    return qpos_start, qpos_end, qvel_start, qvel_end
