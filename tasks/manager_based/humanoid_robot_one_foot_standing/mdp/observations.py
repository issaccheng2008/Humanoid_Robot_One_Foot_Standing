"""Policy observations in the canonical right-support coordinate system."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg

from .symmetry import (
    canonicalize_axial_vector,
    canonicalize_joint_data,
    canonicalize_polar_vector,
)


def _support_is_left(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, 1] > 0.5


def lift_one_foot_in_the_air(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Return only the policy-visible binary lift command."""

    return env.command_manager.get_command(command_name)[:, :1]


def canonical_imu_lin_acc(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    value = base_mdp.imu_lin_acc(env, asset_cfg)
    return canonicalize_polar_vector(value, _support_is_left(env, command_name))


def canonical_imu_ang_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    value = base_mdp.imu_ang_vel(env, asset_cfg)
    return canonicalize_axial_vector(value, _support_is_left(env, command_name))


def canonical_imu_projected_gravity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    value = base_mdp.imu_projected_gravity(env, asset_cfg)
    return canonicalize_polar_vector(value, _support_is_left(env, command_name))


def canonical_joint_pos_rel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    value = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return canonicalize_joint_data(value, _support_is_left(env, command_name))


def canonical_joint_vel_rel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    value = (
        robot.data.joint_vel[:, asset_cfg.joint_ids]
        - robot.data.default_joint_vel[:, asset_cfg.joint_ids]
    )
    return canonicalize_joint_data(value, _support_is_left(env, command_name))


def canonical_last_action(
    env: ManagerBasedRLEnv, action_name: str
) -> torch.Tensor:
    """Return the previous policy action before physical-side mirroring."""

    action_term = env.action_manager.get_term(action_name)
    return action_term.canonical_actions
