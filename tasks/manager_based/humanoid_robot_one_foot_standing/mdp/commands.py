"""Binary lift command and per-episode support-foot selection."""

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class OneFootStandingCommand(CommandTerm):
    """Sample the lift command while keeping one support side per episode.

    Command column 0 is the policy-visible binary lift command. Column 1 is a
    hidden physical-side selector (0 = right support, 1 = left support) used by
    observations, actions, and rewards to implement canonicalization.
    """

    cfg: "OneFootStandingCommandCfg"

    def __init__(self, cfg: "OneFootStandingCommandCfg", env: ManagerBasedRLEnv):
        if not 0.0 <= cfg.lift_probability <= 1.0:
            raise ValueError("lift_probability must be in [0, 1].")
        if not 0.0 <= cfg.support_left_probability <= 1.0:
            raise ValueError("support_left_probability must be in [0, 1].")
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._command[env_ids, 1] = (
            torch.rand_like(self._command[env_ids, 1])
            < self.cfg.support_left_probability
        ).float()
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self._command[env_ids, 0] = (
            torch.rand_like(self._command[env_ids, 0])
            < self.cfg.lift_probability
        ).float()

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class OneFootStandingCommandCfg(CommandTermCfg):
    """Configuration for :class:`OneFootStandingCommand`."""

    class_type: type = OneFootStandingCommand
    asset_name: str = "robot"
    lift_probability: float = 0.5
    support_left_probability: float = 0.5
