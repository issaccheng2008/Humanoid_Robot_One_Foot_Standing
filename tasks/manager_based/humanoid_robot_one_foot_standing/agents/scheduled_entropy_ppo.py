# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO extensions used by the one-foot-standing task."""

from __future__ import annotations

import math
from typing import Any

from rsl_rl.algorithms import PPO


class ScheduledEntropyPPO(PPO):
    """PPO with a one-time, iteration-based entropy coefficient change.

    ``entropy_coef_change_after_iteration`` counts completed PPO updates. The
    configured initial entropy coefficient is therefore used for the first N
    updates, and ``entropy_coef_after_change`` is used starting with update
    N + 1.
    """

    _CHECKPOINT_KEY = "scheduled_entropy_completed_iterations"

    def __init__(
        self,
        *args: Any,
        entropy_coef_change_after_iteration: int = 1000,
        entropy_coef_after_change: float = 0.001,
        **kwargs: Any,
    ) -> None:
        if isinstance(entropy_coef_change_after_iteration, bool) or not isinstance(
            entropy_coef_change_after_iteration, int
        ):
            raise TypeError("entropy_coef_change_after_iteration must be an integer")
        if entropy_coef_change_after_iteration < 0:
            raise ValueError("entropy_coef_change_after_iteration must be non-negative")
        if not math.isfinite(entropy_coef_after_change) or entropy_coef_after_change < 0.0:
            raise ValueError("entropy_coef_after_change must be finite and non-negative")

        super().__init__(*args, **kwargs)
        self._initial_entropy_coef = self.entropy_coef
        self._entropy_coef_change_after_iteration = entropy_coef_change_after_iteration
        self._entropy_coef_after_change = entropy_coef_after_change
        self._completed_iterations = 0
        self._apply_entropy_coef_schedule()

    def _apply_entropy_coef_schedule(self) -> None:
        """Select the coefficient for the next PPO update."""
        if self._completed_iterations >= self._entropy_coef_change_after_iteration:
            self.entropy_coef = self._entropy_coef_after_change
        else:
            self.entropy_coef = self._initial_entropy_coef

    def update(self, *args: Any, **kwargs: Any):
        """Apply the schedule before each PPO update."""
        self._apply_entropy_coef_schedule()
        result = super().update(*args, **kwargs)
        self._completed_iterations += 1
        return result

    def save(self) -> dict:
        """Persist schedule progress alongside the normal PPO state."""
        state = super().save()
        state[self._CHECKPOINT_KEY] = self._completed_iterations
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore schedule progress, including from older runner checkpoints."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)

        if load_iteration:
            if self._CHECKPOINT_KEY in loaded_dict:
                self._completed_iterations = int(loaded_dict[self._CHECKPOINT_KEY])
            elif "iter" in loaded_dict:
                # Legacy checkpoints do not contain the exact update counter.
                # Their runner iteration is the best compatible approximation.
                self._completed_iterations = int(loaded_dict["iter"])
        else:
            self._completed_iterations = 0

        self._apply_entropy_coef_schedule()
        return load_iteration
