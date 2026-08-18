"""Action mapping from the canonical policy side to the physical robot side."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from .symmetry import mirror_joint_data


class CanonicalJointPositionAction(JointPositionAction):
    """Mirror canonical actions when the selected support foot is physically left."""

    cfg: "CanonicalJointPositionActionCfg"

    def __init__(self, cfg: "CanonicalJointPositionActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._canonical_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

    @property
    def canonical_actions(self) -> torch.Tensor:
        return self._canonical_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._canonical_actions[:] = actions
        physical_actions = actions.clone()
        support_is_left = (
            self._env.command_manager.get_command(self.cfg.command_name)[:, 1]
            > 1
        )
        physical_actions[support_is_left] = mirror_joint_data(
            actions[support_is_left]
        )
        super().process_actions(physical_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._canonical_actions[env_ids] = 0.0


@configclass
class CanonicalJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`CanonicalJointPositionAction`."""

    class_type: type = CanonicalJointPositionAction
    command_name: str = MISSING
