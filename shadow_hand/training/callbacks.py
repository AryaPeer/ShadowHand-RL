from __future__ import annotations

import contextlib
from collections import defaultdict, deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from shadow_hand.training.gates import Check, Milestone


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
        for key, floor, why in checks:
            cur = self._recent_mean(key)
            if cur is None:
                rows.append(("SKIP", key, None, floor, why))
                continue

            if cur >= floor:
                rows.append(("OK", key, cur, floor, why))
            else:
                rows.append(("FAIL", key, cur, floor, why))
                failures.append((key, floor, why))

        return rows, failures

    def _evaluate(self, checks: list[Check], label: str) -> None:
        rows, failures = self.evaluate_checks(checks)
        n = min(self._n_rollouts, self._window)
        lines = [
            f"\n===== MILESTONE GATE @ {self.num_timesteps:,} steps — {label} =====",
            f"  recent mean over last {n} rollout(s):",
        ]

        for tag, key, cur, floor, why in rows:
            if cur is None:
                lines.append(f"  [SKIP] {key:32s}  metric not seen — not gated")
            else:
                lines.append(f"  [{tag:<4}] {key:32s} = {cur:9.4f}   floor>={floor:<8.4g} — {why}")

        if failures:
            lines.append("")
            lines.append(f"  VERDICT: STOP — {len(failures)} metric(s) below floor:")
            for key, floor, why in failures:
                lines.append(f"    - {key} < {floor}  ({why})")
            lines.append("  Checkpoints are under runs/<run>/checkpoints/; resume from the latest.")
            self._stop = True
        else:
            lines.append("")
            lines.append("  VERDICT: PASS — all gated metrics above floor; continuing.")

        print("\n".join(lines), flush=True)
        with contextlib.suppress(Exception):
            self.logger.record("gate/passed", 0.0 if failures else 1.0)
