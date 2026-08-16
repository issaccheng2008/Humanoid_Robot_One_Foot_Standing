"""Canonical left-right transformations for one-foot standing."""

from __future__ import annotations

import torch


NUM_JOINTS = 12


def mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    """Reflect ordered right/left leg data across the sagittal plane.

    Joint order must be the six right-leg joints followed by the corresponding
    six left-leg joints. The supplied URDF's joint axes make every paired joint
    coordinate change sign under this reflection.
    """

    if joint_data.shape[-1] != NUM_JOINTS:
        raise ValueError(
            f"Expected {NUM_JOINTS} joint values, got {joint_data.shape[-1]}."
        )

    right = joint_data[..., :6]
    left = joint_data[..., 6:]
    return -torch.cat((left, right), dim=-1)


def canonicalize_joint_data(
    joint_data: torch.Tensor, support_is_left: torch.Tensor
) -> torch.Tensor:
    """Make every sample look like right-foot support/left-foot swing."""

    canonical = joint_data.clone()
    canonical[support_is_left] = mirror_joint_data(joint_data[support_is_left])
    return canonical


def canonicalize_polar_vector(
    vector: torch.Tensor, support_is_left: torch.Tensor
) -> torch.Tensor:
    """Canonicalize a polar vector: ``[x, y, z] -> [x, -y, z]``."""

    canonical = vector.clone()
    canonical[support_is_left, 1] *= -1.0
    return canonical


def canonicalize_axial_vector(
    vector: torch.Tensor, support_is_left: torch.Tensor
) -> torch.Tensor:
    """Canonicalize an axial vector: ``[x, y, z] -> [-x, y, -z]``."""

    canonical = vector.clone()
    canonical[support_is_left, 0] *= -1.0
    canonical[support_is_left, 2] *= -1.0
    return canonical
