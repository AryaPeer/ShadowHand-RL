from __future__ import annotations

import contextlib
import dataclasses
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


class RewardInfoLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buf: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if not (k.startswith(("reward/", "metrics/"))):
                    continue
                try:
                    self._buf[k].append(float(v))
                except (TypeError, ValueError):
                    continue

        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self._buf.items():
            if vals:
                self.logger.record(f"train/{k}", float(np.mean(vals)))

        self._buf.clear()


Check = tuple[str, float, float, str]
Milestone = tuple[int, list[Check], str]


class MilestoneGateCallback(BaseCallback):
    def __init__(
        self, milestones: list[Milestone], window_rollouts: int = 15, verbose: int = 1
    ) -> None:
        super().__init__(verbose)
        self._milestones = sorted(milestones, key=lambda m: m[0])
        self._idx = 0
        self._window = window_rollouts
        self._roll: dict[str, list[float]] = defaultdict(list)
        self._recent: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_rollouts))
        self._n_rollouts = 0
        self._stop = False

    def _on_training_start(self) -> None:
        while (
            self._idx < len(self._milestones)
            and self.num_timesteps >= self._milestones[self._idx][0]
        ):
            self._idx += 1

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if k.startswith(("metrics/", "reward/")):
                    try:
                        self._roll[k].append(float(v))
                    except (TypeError, ValueError):
                        continue

        return not self._stop

    def _on_rollout_end(self) -> None:
        for k, vals in self._roll.items():
            if vals:
                self._recent[k].append(float(np.mean(vals)))

        self._roll.clear()
        self._n_rollouts += 1

        while (
            self._idx < len(self._milestones)
            and self.num_timesteps >= self._milestones[self._idx][0]
        ):
            _, checks, label = self._milestones[self._idx]
            try:
                self._evaluate(checks, label)
            except Exception as exc:
                print(f"[MilestoneGate] WARNING: gate eval errored ({exc!r}); continuing.")

            self._idx += 1

    def _recent_mean(self, key: str) -> float | None:
        vals = self._recent.get(key)
        return float(np.mean(vals)) if vals else None

    def evaluate_checks(self, checks: list[Check]) -> tuple[list[tuple], list[Check]]:
        rows: list[tuple] = []
        failures: list[Check] = []
        for key, floor, base, why in checks:
            cur = self._recent_mean(key)
            if cur is None:
                rows.append(("SKIP", key, None, floor, base, why))
                continue

            if cur >= floor:
                rows.append(("OK", key, cur, floor, base, why))
            else:
                rows.append(("FAIL", key, cur, floor, base, why))
                failures.append((key, floor, base, why))

        return rows, failures

    def _evaluate(self, checks: list[Check], label: str) -> None:
        rows, failures = self.evaluate_checks(checks)
        n = min(self._n_rollouts, self._window)
        lines = [
            f"\n===== MILESTONE GATE @ {self.num_timesteps:,} steps — {label} =====",
            f"  recent mean over last {n} rollout(s):",
        ]

        for tag, key, cur, floor, base, why in rows:
            if cur is None:
                lines.append(f"  [SKIP] {key:32s}  metric not seen — not gated")
            else:
                lines.append(
                    f"  [{tag:<4}] {key:32s} = {cur:9.4f}   "
                    f"5M={base:<8.4g} floor>={floor:<8.4g} — {why}"
                )

        if failures:
            lines.append("")
            lines.append(f"  VERDICT: STOP — {len(failures)} metric(s) below floor:")
            for key, floor, _base, why in failures:
                lines.append(f"    - {key} < {floor}  ({why})")
            lines.append("  A ~500k-cadence checkpoint exists under runs/<run>/checkpoints/.")
            lines.append(
                "  Premature? resume with resume-{peg,grasp}-mjx. "
                "Otherwise fix the cause and restart."
            )
            self._stop = True
        else:
            lines.append("")
            lines.append("  VERDICT: PASS — all gated metrics above floor; continuing.")

        print("\n".join(lines), flush=True)
        with contextlib.suppress(Exception):
            self.logger.record("gate/passed", 0.0 if failures else 1.0)


def setup_sb3_logger(model, run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))


LR_FLOOR = 1e-4
LR_KL_MARGIN = 4.0


def floor_adaptive_lr(model) -> None:
    adaptive = getattr(model, "adaptive_lr", None)
    if adaptive is None:
        return
    adaptive.min_learning_rate = LR_FLOOR
    adaptive.kl_margin = LR_KL_MARGIN
    print(
        f"Adaptive lr bounded: min={LR_FLOOR:.0e} kl_margin={LR_KL_MARGIN} "
        "(peg run 7 random-walked to a 7.7e-05 geo-mean, 23% of rollouts at the 1e-05 floor)"
    )


def dump_run_config(config: Any, run_dir: Path) -> None:
    path = run_dir / "config.json"
    path.write_text(json.dumps(dataclasses.asdict(config), indent=2, default=str))
    print(f"Run config written to {path}")


def apply_saved_config(config: Any, saved: dict, _path: str = "config") -> None:
    for f in dataclasses.fields(config):
        if f.name not in saved:
            continue
        cur = getattr(config, f.name)
        new = saved[f.name]
        if dataclasses.is_dataclass(cur):
            apply_saved_config(cur, new, f"{_path}.{f.name}")
            continue

        if isinstance(cur, tuple) and isinstance(new, list):
            new = tuple(new)
        elif isinstance(cur, list) and cur and isinstance(cur[0], tuple):
            new = [tuple(x) for x in new]

        if new != cur:
            print(f"[resume] {_path}.{f.name}: default {cur!r} -> saved {new!r}")
        setattr(config, f.name, new)


def load_saved_config(config: Any, model_path: Path) -> None:
    for candidate in (model_path.parent, model_path.parent.parent):
        saved_path = candidate / "config.json"
        if saved_path.exists():
            apply_saved_config(config, json.loads(saved_path.read_text()))
            print(f"[resume] applied saved run config from {saved_path}")
            return

    print(
        "[resume] WARNING: no config.json found next to the model — "
        "resuming with dataclass defaults; if the original run used "
        "non-default settings they are NOT reproduced."
    )


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
