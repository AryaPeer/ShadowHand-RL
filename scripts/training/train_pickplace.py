import argparse

from dexterous_hand.config import MjxPickPlaceTrainConfig
from dexterous_hand.envs.pickplace_env import ShadowHandPickPlaceMjxEnv
from scripts.training._common import run_training

PICKPLACE_GATES = [
    (
        10_000_000,
        [
            (
                "metrics/num_finger_contacts",
                0.50,
                1.331,
                "grip forms at all (peaks here, then falls)",
            ),
            ("reward/lifting", 0.10, 0.476, "cube is being lifted off the table"),
        ],
        "pickplace 10M: grip + lift (transient ladder, both collapse later BY DESIGN)",
    ),
    (
        30_000_000,
        [
            ("reward/placed", 0.30, 0.605, "cube is set down near the goal"),
            ("reward/success", 0.30, 3.665, "goal placements are being held"),
        ],
        "pickplace 30M: place + success emergence",
    ),
]


def train(config: MjxPickPlaceTrainConfig) -> None:
    run_training(
        config=config,
        env_cls=ShadowHandPickPlaceMjxEnv,
        run_prefix="pickplace_mjx",
        wandb_name=f"pickplace-mjx-{config.num_envs}env",
        gates=PICKPLACE_GATES,
    )


def parse_args() -> MjxPickPlaceTrainConfig:
    defaults = MjxPickPlaceTrainConfig()
    parser = argparse.ArgumentParser(description="Train Shadow Hand pick-and-place (MJX + SBX PPO)")
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

    return MjxPickPlaceTrainConfig(
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
