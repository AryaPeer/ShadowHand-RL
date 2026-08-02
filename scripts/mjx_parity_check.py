from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig, PickPlaceSceneConfig, SceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.pickplace_scene_builder import build_pickplace_scene
from dexterous_hand.envs.scene_builder import (
    build_scene,
)

GRASP_LIFT_BAR = 0.15
PEG_SETTLE_BAR = 0.73
PEG_HOLD_BAR = 0.70
PICKPLACE_PLACE_BAR = 0.05
PICKPLACE_Z_BAR = 0.02


class CpuEngine:
    name = "cpu"

    def __init__(self, model: mujoco.MjModel, frame_skip: int) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.frame_skip = frame_skip

    def set_state(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def ctrl_step(self, ctrl: np.ndarray, n: int = 1) -> None:
        for _ in range(n):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def xpos(self, body_id: int) -> np.ndarray:
        return np.array(self.data.xpos[body_id])

    def xpos_all(self) -> np.ndarray:
        return np.array(self.data.xpos)

    def xmat_all(self) -> np.ndarray:
        return np.array(self.data.xmat).reshape(self.model.nbody, 9)

    def site_xpos(self, site_id: int) -> np.ndarray:
        return np.array(self.data.site_xpos[site_id])

    def sensordata(self) -> np.ndarray:
        return np.array(self.data.sensordata)


class MjxEngine:
    name = "mjx"

    def __init__(self, model: mujoco.MjModel, frame_skip: int) -> None:
        import jax
        import jax.numpy as jnp
        import mujoco.mjx as mjx

        self._jnp = jnp
        self._mjx = mjx
        self.model = model
        self.frame_skip = frame_skip
        self.mjx_model = mjx.put_model(model)
        self.data = mjx.make_data(self.mjx_model)

        mjx_model = self.mjx_model

        @jax.jit
        def _run(data, ctrl):
            data = data.replace(ctrl=ctrl)

            def sub(d, _):
                return mjx.step(mjx_model, d), None

            data, _ = jax.lax.scan(sub, data, None, length=frame_skip)
            return data

        self._stepper = _run

    def set_state(self, qpos: np.ndarray) -> None:
        jnp = self._jnp
        self.data = self.data.replace(qpos=jnp.asarray(qpos), qvel=jnp.zeros(self.model.nv))
        self.data = self._mjx.forward(self.mjx_model, self.data)

    def ctrl_step(self, ctrl: np.ndarray, n: int = 1) -> None:
        c = self._jnp.asarray(ctrl)
        for _ in range(n):
            self.data = self._stepper(self.data, c)

    def xpos(self, body_id: int) -> np.ndarray:
        return np.asarray(self.data.xpos[body_id])

    def xpos_all(self) -> np.ndarray:
        return np.asarray(self.data.xpos)

    def xmat_all(self) -> np.ndarray:
        return np.asarray(self.data.xmat).reshape(self.model.nbody, 9)

    def site_xpos(self, site_id: int) -> np.ndarray:
        return np.asarray(self.data.site_xpos[site_id])

    def sensordata(self) -> np.ndarray:
        return np.asarray(self.data.sensordata)


def _act(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _set_ctrl(model: mujoco.MjModel, ctrl: np.ndarray, name: str, target: float) -> None:
    ai = _act(model, name)
    if ai < 0:
        return
    lo, hi = model.actuator_ctrlrange[ai]
    ctrl[ai] = float(np.clip(target, lo, hi))


def _set_qpos_joint(model: mujoco.MjModel, qpos: np.ndarray, name: str, val: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    qpos[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))


CUBE_GRIP_SEED = {
    "sx": 0.115,
    "sy": -0.017,
    "z0": -0.02,
    "j3": 1.0,
    "j12": 0.5,
    "thj5": 0.5,
    "th1": 0.7,
    "squeeze": 1.0,
}


def run_grasp(engine_cls) -> dict[str, float]:
    cfg = SceneConfig()
    model, data, nm = build_scene(cfg)
    eng = engine_cls(model, cfg.frame_skip)
    p = CUBE_GRIP_SEED

    qpos = np.array(data.qpos)
    _set_qpos_joint(model, qpos, "slide_x", p["sx"])
    _set_qpos_joint(model, qpos, "slide_y", p["sy"])
    _set_qpos_joint(model, qpos, "slide_z", p["z0"])

    for j in ("FF", "MF", "RF", "LF"):
        _set_qpos_joint(model, qpos, f"rh_{j}J3", p["j3"])
        _set_qpos_joint(model, qpos, f"rh_{j}J2", p["j12"])
        _set_qpos_joint(model, qpos, f"rh_{j}J1", p["j12"])

    _set_qpos_joint(model, qpos, "rh_THJ5", p["thj5"])
    _set_qpos_joint(model, qpos, "rh_THJ4", 1.2)
    _set_qpos_joint(model, qpos, "rh_THJ2", 0.3)
    _set_qpos_joint(model, qpos, "rh_THJ1", p["th1"])
    obj_z0 = cfg.table_height + cfg.object_half_extent + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [0.075, 0.0, obj_z0]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
    eng.set_state(qpos)

    def grip_ctrl(squeeze: float, z: float) -> np.ndarray:
        ctrl = np.zeros(model.nu)
        _set_ctrl(model, ctrl, "slide_x_act", p["sx"])
        _set_ctrl(model, ctrl, "slide_y_act", p["sy"])
        _set_ctrl(model, ctrl, "slide_z_act", z)
        for an in ("rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"):
            _set_ctrl(model, ctrl, an, p["j3"] + squeeze)
        for an in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
            _set_ctrl(model, ctrl, an, p["j12"] * 2 + squeeze)
        _set_ctrl(model, ctrl, "rh_A_THJ5", p["thj5"])
        _set_ctrl(model, ctrl, "rh_A_THJ4", 1.2)
        _set_ctrl(model, ctrl, "rh_A_THJ2", 0.3)
        _set_ctrl(model, ctrl, "rh_A_THJ1", p["th1"] + squeeze)
        return ctrl

    for step in range(30):
        eng.ctrl_step(grip_ctrl(p["squeeze"] * min(step / 10.0, 1.0), p["z0"]))

    for step in range(40):
        t = step / 40.0
        eng.ctrl_step(grip_ctrl(p["squeeze"], p["z0"] + (0.18 - p["z0"]) * t))

    for _ in range(80):
        eng.ctrl_step(grip_ctrl(p["squeeze"], 0.18))

    final_lift = float(eng.xpos(nm.object_body_id)[2] - obj_z0)
    touch = eng.sensordata()[np.asarray(nm.sensor_map.finger_touch_adr)]
    nfc = int(np.sum(touch > 0.0))
    return {"final_lift": final_lift, "nfc_end": float(nfc)}


def run_pickplace(engine_cls) -> dict[str, float]:
    cfg = PickPlaceSceneConfig()
    model, data, nm = build_pickplace_scene(cfg)
    eng = engine_cls(model, cfg.frame_skip)
    p = CUBE_GRIP_SEED
    source_x, source_y = 0.075, 0.0
    goal_x, goal_y = cfg.goal_nominal_xy
    carry_sy = p["sy"] + (goal_y - source_y)

    qpos = np.array(data.qpos)
    _set_qpos_joint(model, qpos, "slide_x", p["sx"])
    _set_qpos_joint(model, qpos, "slide_y", p["sy"])
    _set_qpos_joint(model, qpos, "slide_z", p["z0"])

    for j in ("FF", "MF", "RF", "LF"):
        _set_qpos_joint(model, qpos, f"rh_{j}J3", p["j3"])
        _set_qpos_joint(model, qpos, f"rh_{j}J2", p["j12"])
        _set_qpos_joint(model, qpos, f"rh_{j}J1", p["j12"])

    _set_qpos_joint(model, qpos, "rh_THJ5", p["thj5"])
    _set_qpos_joint(model, qpos, "rh_THJ4", 1.2)
    _set_qpos_joint(model, qpos, "rh_THJ2", 0.3)
    _set_qpos_joint(model, qpos, "rh_THJ1", p["th1"])
    obj_z0 = cfg.table_height + cfg.object_half_extent + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [source_x, source_y, obj_z0]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
    eng.set_state(qpos)

    def grip_ctrl(squeeze: float, z: float, sy: float, open_frac: float = 0.0) -> np.ndarray:
        ctrl = np.zeros(model.nu)
        _set_ctrl(model, ctrl, "slide_x_act", p["sx"])
        _set_ctrl(model, ctrl, "slide_y_act", sy)
        _set_ctrl(model, ctrl, "slide_z_act", z)
        keep = 1.0 - open_frac
        j3 = p["j3"] * keep
        j0 = p["j12"] * 2 * keep
        sq = squeeze * keep
        for an in ("rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"):
            _set_ctrl(model, ctrl, an, j3 + sq)
        for an in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
            _set_ctrl(model, ctrl, an, j0 + sq)
        _set_ctrl(model, ctrl, "rh_A_THJ5", p["thj5"])
        _set_ctrl(model, ctrl, "rh_A_THJ4", 1.2)
        _set_ctrl(model, ctrl, "rh_A_THJ2", 0.3)
        _set_ctrl(model, ctrl, "rh_A_THJ1", (p["th1"] + sq) * keep)
        return ctrl

    z_lift = p["z0"] + 0.12
    for step in range(30):
        eng.ctrl_step(grip_ctrl(p["squeeze"] * min(step / 10.0, 1.0), p["z0"], p["sy"]))

    for step in range(40):
        t = step / 40.0
        eng.ctrl_step(grip_ctrl(p["squeeze"], p["z0"] + (z_lift - p["z0"]) * t, p["sy"]))

    for step in range(50):
        t = step / 50.0
        eng.ctrl_step(grip_ctrl(p["squeeze"], z_lift, p["sy"] + (carry_sy - p["sy"]) * t))

    for step in range(40):
        t = step / 40.0
        eng.ctrl_step(grip_ctrl(p["squeeze"], z_lift + (p["z0"] - z_lift) * t, carry_sy))

    for step in range(20):
        t = step / 20.0
        eng.ctrl_step(grip_ctrl(p["squeeze"], p["z0"], carry_sy, open_frac=min(t * 2.0, 1.0)))

    for _ in range(30):
        eng.ctrl_step(grip_ctrl(0.0, p["z0"] + 0.10, carry_sy, open_frac=1.0))
    for _ in range(40):
        eng.ctrl_step(grip_ctrl(0.0, p["z0"] + 0.10, carry_sy, open_frac=1.0))

    obj = eng.xpos(nm.object_body_id)
    place_dist = float(np.linalg.norm(obj[:2] - np.array([goal_x, goal_y])))
    z_err = float(abs(obj[2] - (cfg.table_height + cfg.object_half_extent)))
    return {"place_dist": place_dist, "obj_z_err": z_err}


def run_peg(engine_cls) -> dict[str, float]:
    import jax.numpy as jnp

    from dexterous_hand.utils.mjx_helpers import get_insertion_depth_jax

    cfg = PegSceneConfig()
    model, data, nm = build_peg_scene(cfg)
    eng = engine_cls(model, cfg.frame_skip)
    peg_len = cfg.peg_half_length * 2.0

    entrance_z = float(model.body("hole").pos[2])
    hx, hy = cfg.hole_offset
    peg_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    ]

    qpos = np.array(data.qpos)
    qpos[peg_qadr : peg_qadr + 3] = [hx, hy, entrance_z + 0.02]
    qpos[peg_qadr + 3 : peg_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    eng.set_state(qpos)

    def depth() -> float:
        return float(
            get_insertion_depth_jax(
                jnp.asarray(eng.xpos_all()),
                jnp.asarray(eng.xmat_all()),
                nm.peg_body_id,
                nm.hole_body_id,
                cfg.peg_half_length,
                cfg.peg_radius,
                cfg.peg_radius + cfg.clearance,
                cfg.hole_depth,
            )
        )

    zero = np.zeros(model.nu)
    for _ in range(25):
        eng.ctrl_step(zero)
    settled = depth() / peg_len
    fracs = []
    for _ in range(30):
        eng.ctrl_step(zero)
        fracs.append(depth() / peg_len)

    return {"settled_frac": settled, "min_hold_frac": float(np.min(fracs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        choices=["both", "cpu", "mjx"],
        default="both",
        help="which engine(s) to run (default both — the parity A/B)",
    )
    args = ap.parse_args()

    engines: list[type[CpuEngine] | type[MjxEngine]] = []
    if args.backend in ("both", "cpu"):
        engines.append(CpuEngine)

    if args.backend in ("both", "mjx"):
        try:
            import mujoco.mjx  # noqa: F401
        except ImportError:
            print(
                "ERROR: mujoco.mjx not importable — install the mjx extra "
                "(uv sync --extra mjx). Refusing to 'pass' without testing MJX."
            )
            raise SystemExit(2) from None

        engines.append(MjxEngine)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for eng_cls in engines:
        for task, fn in (("grasp", run_grasp), ("peg", run_peg), ("pickplace", run_pickplace)):
            t0 = time.time()
            print(f"[{eng_cls.name}] {task} trajectory ...", flush=True)
            r = fn(eng_cls)
            r["seconds"] = time.time() - t0
            results.setdefault(task, {})[eng_cls.name] = r

    print("\n=== parity results ===")
    ok = True
    for task, per_engine in results.items():
        for name, r in per_engine.items():
            if task == "grasp":
                passed = r["final_lift"] >= GRASP_LIFT_BAR and r["nfc_end"] >= 2
                print(
                    f"  grasp     [{name}]: final_lift={r['final_lift'] * 1000:6.1f}mm "
                    f"(bar {GRASP_LIFT_BAR * 1000:.0f}) nfc_end={int(r['nfc_end'])} "
                    f"({r['seconds']:.0f}s)  {'PASS' if passed else 'FAIL'}"
                )
            elif task == "peg":
                passed = r["settled_frac"] >= PEG_SETTLE_BAR and r["min_hold_frac"] >= PEG_HOLD_BAR
                print(
                    f"  peg       [{name}]: settled={r['settled_frac']:.3f} "
                    f"(bar {PEG_SETTLE_BAR}) min_hold={r['min_hold_frac']:.3f} "
                    f"(bar {PEG_HOLD_BAR}) ({r['seconds']:.0f}s)  "
                    f"{'PASS' if passed else 'FAIL'}"
                )
            else:
                passed = (
                    r["place_dist"] <= PICKPLACE_PLACE_BAR and r["obj_z_err"] <= PICKPLACE_Z_BAR
                )
                print(
                    f"  pickplace [{name}]: place_dist={r['place_dist'] * 1000:6.1f}mm "
                    f"(bar {PICKPLACE_PLACE_BAR * 1000:.0f}) z_err={r['obj_z_err'] * 1000:.1f}mm "
                    f"({r['seconds']:.0f}s)  {'PASS' if passed else 'FAIL'}"
                )

            ok &= passed

    print()

    if ok:
        print(
            "PARITY OK — MJX reproduces the CPU-proven winning trajectories."
            if args.backend == "both"
            else "All trajectories PASS."
        )
    else:
        print("PARITY FAILURE — do NOT launch a sanity/full run until resolved.")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
