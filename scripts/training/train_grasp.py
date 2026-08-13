import argparse

from shadow_hand.config import MjxGraspTrainConfig
from shadow_hand.envs.grasp_env import ShadowHandGraspMjxEnv
from scripts.training._common import run_training

GRASP_GATES = [
    (
        10_000_000,
        [
            ("metrics/num_finger_contacts", 0.83, 1.513, "grip forms and stays formed"),
            ("reward/grasping", 0.24, 0.436, "grasp reward maintained"),
        ],
        "grasp 10M: grip health",
    ),
    (
        30_000_000,
        [
            (
                "reward/lifting",
                0.20,
                0.390,
                "lifting still improving, not collapsed (0.390 at 10M and rising)",
            ),
            (
                "metrics/success_hold_steps",
                0.05,
                0.120,
                "the cube is held at target for consecutive steps (0.120 at 10M)",
            ),
        ],
        "grasp 30M: lift emergence",
    ),
]


def train(config: MjxGraspTrainConfig) -> None:
    run_training(
        config=config,
        env_cls=ShadowHandGraspMjxEnv,
        run_prefix="grasp_mjx",
        wandb_name=f"grasp-mjx-{config.num_envs}env",
        gates=GRASP_GATES,
    )


def parse_args() -> MjxGraspTrainConfig:
    defaults = MjxGraspTrainConfig()
    parser = argparse.ArgumentParser(description="Train Shadow Hand grasping (MJX + SBX PPO)")
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
    parser.add_argument(
        "--curriculum-schedule-timesteps",
        type=int,
        default=defaults.curriculum_schedule_timesteps,
        help="Scale the curriculum as if the run were this long. Set it to the long run's "
        "length when sanity-running, so metrics read at step N are comparable to it.",
    )
    args = parser.parse_args()

    return MjxGraspTrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        curriculum_schedule_timesteps=args.curriculum_schedule_timesteps,
        gate_enabled=not args.no_gate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps_per_env=args.n_steps_per_env,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
