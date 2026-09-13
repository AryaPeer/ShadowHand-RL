import pytest

TASKS = ("grasp", "peg", "pickplace")

FILE_TASK = {
    "test_grasp_env.py": "grasp",
    "test_peg_env.py": "peg",
    "test_pickplace_env.py": "pickplace",
    "test_pickplace_reward.py": "pickplace",
    "test_reward_exploits.py": "peg",
}

TASK_KEYWORDS = {
    "grasp": ("grasp", "cube"),
    "peg": ("peg", "tube", "bore", "insert", "hole", "wall_touch", "carry_floor"),
    "pickplace": ("pickplace",),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        owner = FILE_TASK.get(item.path.name)
        if owner is not None:
            tasks = {owner}
        else:
            nid = item.nodeid.lower()
            tasks = {t for t, kws in TASK_KEYWORDS.items() if any(k in nid for k in kws)}

        if not tasks:
            tasks = set(TASKS)

        for t in tasks:
            item.add_marker(getattr(pytest.mark, t))
