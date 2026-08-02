from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np


def _slerp(qa: np.ndarray, qb: np.ndarray, a: float) -> np.ndarray:
    qa = qa / (np.linalg.norm(qa) + 1e-12)
    qb = qb / (np.linalg.norm(qb) + 1e-12)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb, dot = -qb, -dot

    if dot > 0.9995:
        out = qa + a * (qb - qa)
        return out / (np.linalg.norm(out) + 1e-12)

    theta = np.arccos(dot)
    s = np.sin(theta)
    return (np.sin((1.0 - a) * theta) / s) * qa + (np.sin(a * theta) / s) * qb


def _interp_qpos(qa: np.ndarray, qb: np.ndarray, a: float, free_adrs: list[int]) -> np.ndarray:
    q = (1.0 - a) * qa + a * qb
    for adr in free_adrs:
        q[adr + 3 : adr + 7] = _slerp(qa[adr + 3 : adr + 7], qb[adr + 3 : adr + 7], a)
    return q


def _render_task(
    task: str,
    model_path: Path,
    vec_normalize_path: Path,
    out_path: Path,
    steps: int | None,
    seed: int,
    p_pre_grasped: float,
    fps: int = 25,
    seconds: float = 8.0,
    peg_carry_floor: float | None = None,
    tail_steps: int = 15,
) -> dict[str, float]:
    from sbx import PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    from scripts.training._common import load_saved_config

    config_cls: Any
    env_cls: Any
    build_fn: Any
    if task == "grasp":
        from dexterous_hand.config import MjxGraspTrainConfig
        from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv
        from dexterous_hand.envs.scene_builder import build_scene

        config_cls, env_cls, build_fn = MjxGraspTrainConfig, ShadowHandGraspMjxEnv, build_scene
    elif task == "peg":
        from dexterous_hand.config import MjxPegTrainConfig
        from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv
        from dexterous_hand.envs.peg_scene_builder import build_peg_scene

        config_cls, env_cls, build_fn = MjxPegTrainConfig, ShadowHandPegMjxEnv, build_peg_scene
    else:
        from dexterous_hand.config import MjxPickPlaceTrainConfig
        from dexterous_hand.envs.pickplace_env import ShadowHandPickPlaceMjxEnv
        from dexterous_hand.envs.pickplace_scene_builder import build_pickplace_scene

        config_cls, env_cls, build_fn = (
            MjxPickPlaceTrainConfig,
            ShadowHandPickPlaceMjxEnv,
            build_pickplace_scene,
        )

    config = config_cls()
    load_saved_config(config, model_path)
    config.num_envs = 1
    config.seed = seed
    config.obs_noise_std = 0.0

    if steps is None:
        steps = config.max_episode_steps

    raw_env: Any = env_cls.from_config(config)

    if task == "peg":
        cf = (
            config.adaptive_curriculum.carry_floor_levels[-1]
            if peg_carry_floor is None
            else peg_carry_floor
        )
        raw_env.set_curriculum_params(
            clearance=config.adaptive_curriculum.clearance,
            carry_floor=cf,
        )
    elif task == "grasp" and config.curriculum_stages:
        raw_env.set_curriculum_params(p_pre_grasped=p_pre_grasped)

    env: Any = VecMonitor(raw_env)
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    cpu_model, cpu_data, _nm = build_fn(config.scene_config)
    renderer = mujoco.Renderer(cpu_model, height=480, width=640)

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    obs = env.reset()
    qpos_traj: list[np.ndarray] = []
    mocap_traj: list[np.ndarray] = []
    succ_step: int | None = None
    for t in range(steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, _rewards, _dones, infos = env.step(actions)
        for k, v in infos[0].items():
            if k.startswith(("metrics/", "reward/")):
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                metric_counts[k] = metric_counts.get(k, 0) + 1

        if succ_step is None and float(infos[0].get("reward/success", 0.0)) > 0.0:
            succ_step = t
        qpos_traj.append(np.asarray(raw_env._mjx_data_batch.qpos[0]).copy())
        if cpu_model.nmocap > 0:
            mocap_traj.append(np.asarray(raw_env._mjx_data_batch.mocap_pos[0]).copy())

    end = min(len(qpos_traj), (succ_step + tail_steps) if succ_step is not None else 150)
    qpos_traj = qpos_traj[:end]
    mocap_traj = mocap_traj[:end]

    free_adrs = [
        int(cpu_model.jnt_qposadr[j])
        for j in range(cpu_model.njnt)
        if int(cpu_model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    src = max(len(qpos_traj), 1)
    n_out = max(round(seconds * fps), src)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for i in range(n_out):
        s = i * (src - 1) / max(n_out - 1, 1)
        lo = int(np.floor(s))
        hi = min(lo + 1, src - 1)
        a = s - lo
        cpu_data.qpos[:] = _interp_qpos(qpos_traj[lo], qpos_traj[hi], a, free_adrs)
        if cpu_model.nmocap > 0 and mocap_traj:
            cpu_data.mocap_pos[:] = (1.0 - a) * mocap_traj[lo] + a * mocap_traj[hi]
        mujoco.mj_forward(cpu_model, cpu_data)
        renderer.update_scene(cpu_data, camera="track_cam")
        writer.append_data(renderer.render().copy())

    writer.close()
    renderer.close()

    return {k: metric_sums[k] / metric_counts[k] for k in sorted(metric_sums)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a deterministic rollout of a trained checkpoint to mp4."
    )
    ap.add_argument("--task", choices=("grasp", "peg", "pickplace", "both"), default="both")
    ap.add_argument("--grasp-model", type=Path, default=None)
    ap.add_argument("--grasp-vec-normalize", type=Path, default=None)
    ap.add_argument("--peg-model", type=Path, default=None)
    ap.add_argument("--peg-vec-normalize", type=Path, default=None)
    ap.add_argument("--pickplace-model", type=Path, default=None)
    ap.add_argument("--pickplace-vec-normalize", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/render_overnight"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-pre-grasped", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="target minimum clip length; trajectory is smoothly resampled to "
        "max(seconds*fps, len) frames (never sped up)",
    )
    ap.add_argument(
        "--peg-carry-floor",
        type=float,
        default=None,
        help="gripped carry distance to render; default = full carry (top level)",
    )
    ap.add_argument(
        "--tail-steps",
        type=int,
        default=15,
        help="frames kept after first success; lower cuts more post-success idle fidgeting",
    )
    args = ap.parse_args()

    tasks = ["grasp", "peg"] if args.task == "both" else [args.task]
    model_by_task = {
        "grasp": (args.grasp_model, args.grasp_vec_normalize),
        "peg": (args.peg_model, args.peg_vec_normalize),
        "pickplace": (args.pickplace_model, args.pickplace_vec_normalize),
    }

    for task in tasks:
        model_path, vn_path = model_by_task[task]
        if model_path is None or vn_path is None:
            raise SystemExit(
                f"--{task}-model and --{task}-vec-normalize are required for task={task}"
            )

        out_path = args.out_dir / f"{task}_rollout.mp4"
        print(
            f"[{task}] rendering deterministic rollout (p_pre_grasped="
            f"{args.p_pre_grasped:.2f}) -> {out_path}",
            flush=True,
        )
        summary = _render_task(
            task,
            model_path,
            vn_path,
            out_path,
            args.steps,
            args.seed,
            args.p_pre_grasped,
            fps=args.fps,
            seconds=args.seconds,
            peg_carry_floor=args.peg_carry_floor,
            tail_steps=args.tail_steps,
        )
        print(f"[{task}] done. per-step metric means over the rollout:")
        for k, v in summary.items():
            print(f"    {k:36s} = {v:.4f}")


if __name__ == "__main__":
    main()
