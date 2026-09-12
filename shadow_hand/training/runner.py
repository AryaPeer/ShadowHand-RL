from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from shadow_hand.training.callbacks import MilestoneGateCallback, RewardInfoLoggerCallback
from shadow_hand.training.config_io import dump_run_config
from shadow_hand.training.gates import Milestone


def setup_sb3_logger(model: Any, run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))


LR_FLOOR = 1e-4
LR_KL_MARGIN = 4.0


def floor_adaptive_lr(model: Any) -> None:
    adaptive = getattr(model, "adaptive_lr", None)
    if adaptive is None:
        return
    adaptive.min_learning_rate = LR_FLOOR
    adaptive.kl_margin = LR_KL_MARGIN
    print(f"Adaptive lr bounded: min={LR_FLOOR:.0e} kl_margin={LR_KL_MARGIN}")


def run_training(
    *,
    config: Any,
    env_cls: Any,
    run_prefix: str,
    wandb_name: str,
    gates: list[Milestone],
    extra_callbacks: list[BaseCallback] | None = None,
    extra_wandb_config: dict[str, Any] | None = None,
) -> None:
    import flax.linen as nn
    import wandb
    from sbx import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
    from wandb.integration.sb3 import WandbCallback

    from shadow_hand.policies.clamped_actor import make_clamped_actor

    run_dir = Path("runs") / f"{run_prefix}_{config.num_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rollout_size = config.num_envs * config.n_steps_per_env
    if config.batch_size > rollout_size:
        new_bs = max(rollout_size // 4, 64)
        print(
            f"WARNING: batch_size {config.batch_size} > rollout_size {rollout_size}. "
            f"Auto-resized to {new_bs}."
        )
        config.batch_size = new_bs

    dump_run_config(config, run_dir)

    wandb_config = dataclasses.asdict(config)
    if extra_wandb_config:
        wandb_config.update(extra_wandb_config)
    wandb.init(project="shadow-hand", name=wandb_name, config=wandb_config)

    vec_env = env_cls.from_config(config)
    vec_env = VecMonitor(vec_env)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config.gamma,
        )

    activation_fn = {"elu": nn.elu, "relu": nn.relu, "tanh": nn.tanh}[config.activation]

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps_per_env,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        policy_kwargs={
            "net_arch": {"pi": config.net_arch.copy(), "vf": config.net_arch.copy()},
            "activation_fn": activation_fn,
            "log_std_init": config.log_std_init,
            "actor_class": make_clamped_actor(
                log_std_min=config.log_std_min,
                log_std_max=config.log_std_max,
            ),
        },
        verbose=1,
        seed=config.seed,
    )

    floor_adaptive_lr(model)
    setup_sb3_logger(model, run_dir)

    callbacks: list[BaseCallback] = list(extra_callbacks or [])
    if config.gate_enabled:
        callbacks.append(MilestoneGateCallback(gates, verbose=1))
    callbacks += [
        RewardInfoLoggerCallback(),
        CheckpointCallback(
            save_freq=max(500_000 // config.num_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            save_vecnormalize=True,
        ),
        WandbCallback(
            model_save_path=str(run_dir),
            model_save_freq=max(100_000 // config.num_envs, 1),
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Saved to {run_dir}")
    wandb.finish()
    vec_env.close()


def run_resume(
    *,
    args: Any,
    config: Any,
    env_cls: Any,
    extra_callbacks: list[BaseCallback] | None = None,
) -> None:
    from sbx import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    model_path = Path(args.model_path).expanduser().resolve()
    vec_norm_path = Path(args.vec_normalize_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path}")
    if not vec_norm_path.exists():
        raise FileNotFoundError(f"vec_normalize not found at {vec_norm_path}")

    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        src = model_path.parent
        run_dir = src.with_name(src.name + "_resumed")

    run_dir.mkdir(parents=True, exist_ok=True)

    dump_run_config(config, run_dir)

    vec_env = env_cls.from_config(config)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
    vec_env.training = True
    vec_env.norm_reward = config.norm_reward

    model = PPO.load(str(model_path), env=vec_env)
    model.target_kl = config.target_kl

    floor_adaptive_lr(model)
    setup_sb3_logger(model, run_dir)

    callbacks: list[BaseCallback] = list(extra_callbacks or [])
    callbacks += [
        RewardInfoLoggerCallback(),
        CheckpointCallback(
            save_freq=max(500_000 // config.num_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            save_vecnormalize=True,
        ),
    ]

    print(f"Resuming from {model_path} for {args.additional_timesteps:,} additional timesteps.")
    print(f"Output dir: {run_dir}")

    model.learn(
        total_timesteps=args.additional_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(str(run_dir / "final_model"))
    vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Saved to {run_dir}")
    vec_env.close()
