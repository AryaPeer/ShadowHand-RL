from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TASK_NAMES: tuple[str, ...] = ("grasp", "peg", "pickplace")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    config_cls: Any
    env_cls: Any
    build_scene: Any


def load_task(name: str) -> TaskSpec:
    if name == "grasp":
        from shadow_hand.config import GraspTrainConfig
        from shadow_hand.envs.grasp import GraspEnv
        from shadow_hand.scenes.grasp import build_grasp_scene

        return TaskSpec(name, GraspTrainConfig, GraspEnv, build_grasp_scene)

    if name == "peg":
        from shadow_hand.config import PegTrainConfig
        from shadow_hand.envs.peg import PegEnv
        from shadow_hand.scenes.peg import build_peg_scene

        return TaskSpec(name, PegTrainConfig, PegEnv, build_peg_scene)

    if name == "pickplace":
        from shadow_hand.config import PickPlaceTrainConfig
        from shadow_hand.envs.pickplace import PickPlaceEnv
        from shadow_hand.scenes.pickplace import build_pickplace_scene

        return TaskSpec(name, PickPlaceTrainConfig, PickPlaceEnv, build_pickplace_scene)

    raise ValueError(f"unknown task: {name!r} (expected one of {', '.join(TASK_NAMES)})")
