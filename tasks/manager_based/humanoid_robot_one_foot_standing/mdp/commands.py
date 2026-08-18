"""Scheduled binary lift command and per-episode support-foot selection."""

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class OneFootStandingCommand(CommandTerm):
    """Run a stand, lift, then lower sequence with one support side per episode.

    Command column 0 is the policy-visible binary lift command. Column 1 is a
    hidden physical-side selector (0 = right support, 1 = left support) used by
    observations, actions, and rewards to implement canonicalization. When
    switching is disabled, column 1 remains zero so every subsystem uses the
    physical right foot as support without any mirroring.
    """

    cfg: "OneFootStandingCommandCfg"

    def __init__(self, cfg: "OneFootStandingCommandCfg", env: ManagerBasedRLEnv):
        for name, time_range in (
            ("initial_stand_time_range_s", cfg.initial_stand_time_range_s),
            ("lift_time_range_s", cfg.lift_time_range_s),
        ):
            if time_range[0] <= 0.0 or time_range[1] < time_range[0]:
                raise ValueError(f"{name} must be a positive, ordered range.")
        if not 0.0 <= cfg.support_left_probability <= 1.0:
            raise ValueError("support_left_probability must be in [0, 1].")
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._initial_stand_duration_s = torch.zeros(
            self.num_envs, device=self.device
        )
        self._lift_duration_s = torch.zeros(self.num_envs, device=self.device)
        self._time_since_lift_ended_s = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def time_since_lift_ended_s(self) -> torch.Tensor:
        """Time since the lift command most recently changed from one to zero."""

        return self._time_since_lift_ended_s

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_resets = len(env_ids)
        self._initial_stand_duration_s[env_ids] = torch.empty(
            num_resets, device=self.device
        ).uniform_(*self.cfg.initial_stand_time_range_s)
        self._lift_duration_s[env_ids] = torch.empty(
            num_resets, device=self.device
        ).uniform_(*self.cfg.lift_time_range_s)
        if self.cfg.enable_left_right_switching:
            self._command[env_ids, 1] = (
                torch.rand_like(self._command[env_ids, 1])
                < self.cfg.support_left_probability
            ).float()
        else:
            # Canonical physical side: right support (index 0), left swing.
            self._command[env_ids, 1] = 0.0
        self._command[env_ids, 0] = 0.0
        # The initial command-zero phase should retain the default-pose reward.
        self._time_since_lift_ended_s[env_ids] = torch.inf
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self._command[env_ids, 0] = 0.0

    def _update_command(self) -> None:
        elapsed_time_s = self._env.episode_length_buf * self._env.step_dt
        lift_start_s = self._initial_stand_duration_s
        lift_end_s = lift_start_s + self._lift_duration_s
        previous_lift_command = self._command[:, 0] > 0.5
        lift_command = (
            (elapsed_time_s >= lift_start_s) & (elapsed_time_s < lift_end_s)
        )
        lift_just_ended = previous_lift_command & ~lift_command
        self._time_since_lift_ended_s = torch.where(
            lift_command,
            torch.zeros_like(self._time_since_lift_ended_s),
            torch.where(
                lift_just_ended,
                torch.zeros_like(self._time_since_lift_ended_s),
                self._time_since_lift_ended_s + self._env.step_dt,
            ),
        )
        self._command[:, 0] = lift_command.float()

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
    initial_stand_time_range_s: tuple[float, float] = (1.0, 2.0)
    lift_time_range_s: tuple[float, float] = (3.0, 5.0)
    enable_left_right_switching: bool = True
    support_left_probability: float = 0.5
