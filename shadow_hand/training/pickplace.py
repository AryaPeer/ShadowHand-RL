from __future__ import annotations

from pathlib import Path

from shadow_hand.config import PickPlaceTrainConfig
from shadow_hand.envs.pickplace import PickPlaceEnv
from shadow_hand.training.args import (
    build_resume_parser,
    build_train_parser,
    train_config_kwargs,
)
from shadow_hand.training.config_io import load_saved_config
from shadow_hand.training.gates import PICKPLACE_GATES
from shadow_hand.training.runner import run_resume, run_training


def train(argv: list[str] | None = None) -> None:
    defaults = PickPlaceTrainConfig()
    parser = build_train_parser("Train Shadow Hand pick-and-place (MJX + SBX PPO)", defaults)
    args = parser.parse_args(argv)

    config = PickPlaceTrainConfig(**train_config_kwargs(args))
    run_training(
        config=config,
        env_cls=PickPlaceEnv,
        run_prefix="pickplace",
        wandb_name=f"pickplace-{config.num_envs}env",
        gates=PICKPLACE_GATES,
    )


def resume(argv: list[str] | None = None) -> None:
    args = build_resume_parser("Resume Shadow Hand pick-and-place (MJX + SBX PPO)").parse_args(argv)

    config = PickPlaceTrainConfig()
    load_saved_config(config, Path(args.model_path).expanduser().resolve())
    config.num_envs = args.num_envs
    config.seed = args.seed

    run_resume(args=args, config=config, env_cls=PickPlaceEnv)
