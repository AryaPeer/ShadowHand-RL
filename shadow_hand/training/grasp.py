from __future__ import annotations

from pathlib import Path

from shadow_hand.config import GraspTrainConfig
from shadow_hand.envs.grasp import GraspEnv
from shadow_hand.training.args import (
    build_resume_parser,
    build_train_parser,
    train_config_kwargs,
)
from shadow_hand.training.config_io import load_saved_config
from shadow_hand.training.gates import GRASP_GATES
from shadow_hand.training.runner import run_resume, run_training


def train(argv: list[str] | None = None) -> None:
    defaults = GraspTrainConfig()
    parser = build_train_parser("Train Shadow Hand grasping (MJX + SBX PPO)", defaults)
    parser.add_argument(
        "--curriculum-schedule-timesteps",
        type=int,
        default=defaults.curriculum_schedule_timesteps,
        help="Scale the curriculum as if the run were this long.",
    )
    args = parser.parse_args(argv)

    config = GraspTrainConfig(
        curriculum_schedule_timesteps=args.curriculum_schedule_timesteps,
        **train_config_kwargs(args),
    )
    run_training(
        config=config,
        env_cls=GraspEnv,
        run_prefix="grasp",
        wandb_name=f"grasp-{config.num_envs}env",
        gates=GRASP_GATES,
    )


def resume(argv: list[str] | None = None) -> None:
    args = build_resume_parser("Resume Shadow Hand grasping (MJX + SBX PPO)").parse_args(argv)

    config = GraspTrainConfig()
    load_saved_config(config, Path(args.model_path).expanduser().resolve())
    config.num_envs = args.num_envs
    config.seed = args.seed

    run_resume(args=args, config=config, env_cls=GraspEnv)
