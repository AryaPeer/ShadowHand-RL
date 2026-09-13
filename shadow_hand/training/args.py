from __future__ import annotations

import argparse
from typing import Any


def build_train_parser(description: str, defaults: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--n-steps-per-env", type=int, default=defaults.n_steps_per_env)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable the milestone compute-saver gate (let the run go to the end).",
    )
    return parser


def train_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "gate_enabled": not args.no_gate,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_steps_per_env": args.n_steps_per_env,
        "seed": args.seed,
    }


def build_resume_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
    return parser
