from __future__ import annotations

from typing import Any


def apply_eval_curriculum(
    env: Any,
    task: str,
    config: Any,
    *,
    p_pre_grasped: float = 0.0,
    peg_carry_floor: float | None = None,
) -> None:
    if task == "peg":
        kwargs: dict[str, float] = {"clearance": config.adaptive_curriculum.clearance}
        if peg_carry_floor is not None:
            kwargs["carry_floor"] = peg_carry_floor
        env.set_curriculum_params(**kwargs)
    elif task == "grasp" and config.curriculum_stages:
        env.set_curriculum_params(p_pre_grasped=p_pre_grasped)
