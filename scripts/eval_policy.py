import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic-policy evaluation of a saved checkpoint."
    )
    parser.add_argument("--task", choices=("grasp", "peg", "pickplace"), required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vec-normalize-path", type=str, required=True)
    parser.add_argument("-n", "--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--p-pre-grasped",
        type=float,
        default=0.0,
        help="Fraction of eval episodes spawned already gripping the object. Default 0.0 "
        "measures the real task; pass the training stage value to reproduce a train-time number.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.p_pre_grasped <= 1.0:
        parser.error("--p-pre-grasped must be in [0, 1]")

    from sbx import PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    from scripts.training._common import load_saved_config

    config_cls: Any
    env_cls: Any
    if args.task == "grasp":
        from dexterous_hand.config import MjxGraspTrainConfig
        from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv

        config_cls, env_cls = MjxGraspTrainConfig, ShadowHandGraspMjxEnv
    elif args.task == "peg":
        from dexterous_hand.config import MjxPegTrainConfig
        from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv

        config_cls, env_cls = MjxPegTrainConfig, ShadowHandPegMjxEnv
    else:
        from dexterous_hand.config import MjxPickPlaceTrainConfig
        from dexterous_hand.envs.pickplace_env import ShadowHandPickPlaceMjxEnv

        config_cls, env_cls = MjxPickPlaceTrainConfig, ShadowHandPickPlaceMjxEnv

    model_path = Path(args.model_path).expanduser().resolve()
    config = config_cls()
    load_saved_config(config, model_path)
    config.num_envs = args.num_envs
    config.seed = args.seed
    config.obs_noise_std = 0.0

    env: Any = env_cls.from_config(config)

    if args.task == "peg":
        env.set_curriculum_params(clearance=config.adaptive_curriculum.clearance)
    elif args.task == "grasp" and config.curriculum_stages:
        env.set_curriculum_params(p_pre_grasped=args.p_pre_grasped)

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


if __name__ == "__main__":
    main()
