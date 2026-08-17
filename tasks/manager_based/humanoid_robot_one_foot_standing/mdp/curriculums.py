# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for one-foot standing training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class modify_reward_param_linearly(ManagerTermBase):
    """Linearly move one numeric reward parameter between two values."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        term_name = cfg.params["term_name"]
        param_name = cfg.params["param_name"]
        start_step = cfg.params["start_step"]
        end_step = cfg.params["end_step"]

        if start_step < 0:
            raise ValueError("start_step must be non-negative.")
        if end_step <= start_step:
            raise ValueError("end_step must be greater than start_step.")

        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        if param_name not in self._term_cfg.params:
            raise ValueError(
                f"Reward term {term_name!r} has no parameter {param_name!r}."
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        param_name: str,
        start_value: float,
        end_value: float,
        start_step: int,
        end_step: int,
    ) -> float:
        del env_ids

        progress = (env.common_step_counter - start_step) / (end_step - start_step)
        progress = max(0.0, min(1.0, progress))
        value = start_value + progress * (end_value - start_value)

        if self._term_cfg.params[param_name] != value:
            self._term_cfg.params[param_name] = value
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return value
