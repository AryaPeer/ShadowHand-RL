from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hand.tasks import TASK_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic-policy evaluation of a saved checkpoint."
    )
    parser.add_argument("task", choices=TASK_NAMES)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vec-normalize-path", type=str, required=True)
    parser.add_argument("-n", "--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--p-pre-grasped",
        type=float,
        default=0.0,
        help="Fraction of eval episodes spawned already gripping the object.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.p_pre_grasped <= 1.0:
        parser.error("--p-pre-grasped must be in [0, 1]")

    from sbx import PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    from shadow_hand.evaluation._curriculum import apply_eval_curriculum
    from shadow_hand.tasks import load_task
    from shadow_hand.training.config_io import load_saved_config

    spec = load_task(args.task)

    model_path = Path(args.model_path).expanduser().resolve()
    config = spec.config_cls()
    load_saved_config(config, model_path)
    config.num_envs = args.num_envs
    config.seed = args.seed
    config.obs_noise_std = 0.0

    env: Any = spec.env_cls.from_config(config)
    apply_eval_curriculum(env, args.task, config, p_pre_grasped=args.p_pre_grasped)

    print(f"[eval] p_pre_grasped={args.p_pre_grasped:.2f}")

    env = VecMonitor(env)
    env = VecNormalize.load(str(Path(args.vec_normalize_path).expanduser().resolve()), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    obs = env.reset()
    completed = 0
    successes = 0
    metric_sums: dict[str, float] = defaultdict(float)
    metric_counts: dict[str, int] = defaultdict(int)
    last_success = np.zeros(args.num_envs)

    while completed < args.episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, _rewards, dones, infos = env.step(actions)
        for i, info in enumerate(infos):
            if "is_success" in info:
                last_success[i] = float(info["is_success"])

            for k, v in info.items():
                if k.startswith(("metrics/", "reward/")):
                    metric_sums[k] += float(v)
                    metric_counts[k] += 1

        for i in np.flatnonzero(dones).tolist():
            completed += 1
            successes += int(last_success[i] > 0.5)
            last_success[i] = 0.0

    print(f"\n=== deterministic eval: {args.task} ===")
    print(f"  episodes completed : {completed}")
    print(f"  success rate       : {successes / completed:.3f}  (is_success at episode end)")
    for k in sorted(metric_sums):
        print(f"  {k:36s} = {metric_sums[k] / metric_counts[k]:.4f}  (per-step mean)")
