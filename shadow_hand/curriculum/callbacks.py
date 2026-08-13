import contextlib
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from shadow_hand.config import AdaptiveCurriculumConfig


class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(
        self,
        cfg: AdaptiveCurriculumConfig,
        success_bonus_per_step: float,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self._success_bonus = float(success_bonus_per_step)
        self._levels = tuple(cfg.carry_floor_levels)
        self._level_idx = 0
        self._carry_floor = self._levels[0]
        self._roll: list[float] = []
        self._recent: deque[float] = deque(maxlen=cfg.success_window_rollouts)
        self._n_rollouts = 0
        self._last_adjust = 0

    def _apply(self, carry_floor: float) -> None:
        self.training_env.env_method(
            "set_curriculum_params",
            self.cfg.clearance,
            carry_floor,
        )

        if self.verbose:
            print(
                f"[AdaptiveCurriculum] carry_floor={carry_floor:.2f} "
                f"clearance={self.cfg.clearance * 1000:.1f}mm at step {self.num_timesteps}",
                flush=True,
            )

    def _on_training_start(self) -> None:
        self._apply(self._carry_floor)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "reward/success" in info:
                try:
                    self._roll.append(float(info["reward/success"]))
                except (TypeError, ValueError):
                    continue

        return True

    def _success_frac(self) -> float:
        if not self._recent or self._success_bonus <= 0:
            return 0.0
        return float(np.mean(self._recent)) / self._success_bonus

    def _on_rollout_end(self) -> None:
        if self._roll:
            self._recent.append(float(np.mean(self._roll)))
            self._roll.clear()

        self._n_rollouts += 1

        frac = self._success_frac()
        with contextlib.suppress(Exception):
            self.logger.record("curriculum/carry_floor", self._carry_floor)
            self.logger.record("curriculum/success_frac", frac)

        ready = (
            len(self._recent) >= (self._recent.maxlen or 0)
            and self._n_rollouts - self._last_adjust >= self.cfg.check_interval_rollouts
        )
        if not ready:
            return

        idx = self._level_idx
        if frac > self.cfg.advance_success_frac:
            idx = min(idx + 1, len(self._levels) - 1)
        elif frac < self.cfg.retreat_success_frac:
            idx = max(idx - 1, 0)

        if idx != self._level_idx:
            self._level_idx = idx
            self._carry_floor = self._levels[idx]
            self._last_adjust = self._n_rollouts
            self._apply(self._carry_floor)
