import argparse

from dexterous_hand.config import MjxPegTrainConfig
from dexterous_hand.curriculum.callbacks import AdaptiveCurriculumCallback
from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv
from scripts.training._common import run_training

_NAN = float("nan")

PEG_GATES = [
    (
        10_000_000,
        [
            (
                "reward/place",
                0.20,
                _NAN,
                "carrying or inserting (run 7 healthy 0.49; a collapsed run sits at 0.011)",
            ),
            ("metrics/stage", 1.0, _NAN, "task progressed past grasp-and-sit"),
        ],
        "peg 10M: ladder acquired (terminal insertion is NOT expected this early)",
    ),
    (
        30_000_000,
        [
            ("reward/place", 0.40, _NAN, "peg reaching the socket, not parked mid-carry"),
            (
                "metrics/insertion_depth",
                0.001,
                _NAN,
                "first in-bore insertion (run 7 noise floor was 3e-4)",
            ),
        ],
        "peg 30M: carry consolidated and insertion starting",
    ),
    (
        40_000_000,
        [
            ("metrics/insertion_depth", 0.003, _NAN, "in-bore insertion happening"),
            ("reward/place_release", 0.03, _NAN, "engaged-release occurring in-bore"),
            ("reward/success", 0.05, _NAN, "some insertion succeeding"),
        ],
        "peg 40M: terminal engaged-release exists",
    ),
]


def train(config: MjxPegTrainConfig) -> None:
    run_training(
        config=config,
        env_cls=ShadowHandPegMjxEnv,
        run_prefix="peg_mjx",
        wandb_name=f"peg-mjx-{config.num_envs}env",
        gates=PEG_GATES,
        extra_callbacks=[
            AdaptiveCurriculumCallback(
                config.adaptive_curriculum,
                config.reward_config.success_bonus_per_step,
                verbose=1,
            )
        ],
    )


def parse_args() -> MjxPegTrainConfig:
    defaults = MjxPegTrainConfig()
    parser = argparse.ArgumentParser(description="Train Shadow Hand peg-in-hole (MJX + SBX PPO)")
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--n-steps-per-env", type=int, default=defaults.n_steps_per_env)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable the 10M/30M milestone compute-saver gate (let the run go to the end).",
    )
    args = parser.parse_args()

    return MjxPegTrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        gate_enabled=not args.no_gate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps_per_env=args.n_steps_per_env,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
