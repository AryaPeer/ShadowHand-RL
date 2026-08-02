import argparse
from pathlib import Path

from dexterous_hand.config import MjxPegTrainConfig
from dexterous_hand.curriculum.callbacks import AdaptiveCurriculumCallback
from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv
from scripts.training._common import load_saved_config, run_resume


def train(args: argparse.Namespace) -> None:
    config = MjxPegTrainConfig()
    load_saved_config(config, Path(args.model_path).expanduser().resolve())
    config.num_envs = args.num_envs
    config.seed = args.seed

    run_resume(
        args=args,
        config=config,
        env_cls=ShadowHandPegMjxEnv,
        extra_callbacks=[
            AdaptiveCurriculumCallback(
                config.adaptive_curriculum,
                config.reward_config.success_bonus_per_step,
                verbose=1,
            )
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume Shadow Hand peg-in-hole (MJX + SBX PPO)")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to final_model.zip (or any checkpoint .zip)",
    )
    parser.add_argument(
        "--vec-normalize-path",
        type=str,
        required=True,
        help="Path to vec_normalize.pkl saved alongside the model",
    )
    parser.add_argument(
        "--additional-timesteps",
        type=int,
        required=True,
        help="How many MORE timesteps to train (additional, not cumulative)",
    )
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save resumed run (default: <input_dir>_resumed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
