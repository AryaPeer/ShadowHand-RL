from __future__ import annotations

from pathlib import Path

from shadow_hand.config import PegTrainConfig
from shadow_hand.curriculum.callbacks import AdaptiveCurriculumCallback
from shadow_hand.envs.peg import PegEnv
from shadow_hand.training.args import (
    build_resume_parser,
    build_train_parser,
    train_config_kwargs,
)
from shadow_hand.training.config_io import load_saved_config
from shadow_hand.training.gates import PEG_GATES
from shadow_hand.training.runner import run_resume, run_training


def _curriculum_callback(config: PegTrainConfig) -> AdaptiveCurriculumCallback:
    return AdaptiveCurriculumCallback(
        config.adaptive_curriculum,
        config.reward_config.success_bonus_per_step,
        verbose=1,
    )


def train(argv: list[str] | None = None) -> None:
    defaults = PegTrainConfig()
    parser = build_train_parser("Train Shadow Hand peg-in-hole (MJX + SBX PPO)", defaults)
    args = parser.parse_args(argv)

    config = PegTrainConfig(**train_config_kwargs(args))
    run_training(
        config=config,
        env_cls=PegEnv,
        run_prefix="peg",
        wandb_name=f"peg-{config.num_envs}env",
        gates=PEG_GATES,
        extra_callbacks=[_curriculum_callback(config)],
    )


def resume(argv: list[str] | None = None) -> None:
    args = build_resume_parser("Resume Shadow Hand peg-in-hole (MJX + SBX PPO)").parse_args(argv)

    config = PegTrainConfig()
    load_saved_config(config, Path(args.model_path).expanduser().resolve())
    config.num_envs = args.num_envs
    config.seed = args.seed

    run_resume(
        args=args,
        config=config,
        env_cls=PegEnv,
        extra_callbacks=[_curriculum_callback(config)],
    )
