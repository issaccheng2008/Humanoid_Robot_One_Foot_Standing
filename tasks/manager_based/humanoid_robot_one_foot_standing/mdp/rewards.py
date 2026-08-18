"""Reward functions for canonical one-foot standing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_TIME_OFF_GROUND_STATE_ATTR = "_one_foot_time_off_ground_state"


def base_linear_velocity_zero(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward a stationary base, including vertical motion."""

    if std <= 0.0:
        raise ValueError("std must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    speed_sq = torch.sum(torch.square(robot.data.root_lin_vel_w), dim=1)
    return torch.exp(-speed_sq / std**2)


def base_angular_velocity_zero(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward zero roll, pitch, and yaw angular velocity."""

    if std <= 0.0:
        raise ValueError("std must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    speed_sq = torch.sum(torch.square(robot.data.root_ang_vel_w), dim=1)
    return torch.exp(-speed_sq / std**2)


def base_acceleration_l2(
    env: ManagerBasedRLEnv, axis: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize squared base acceleration on x/y in yaw frame or z in world."""

    robot: Articulation = env.scene[asset_cfg.name]
    base_acc_w = robot.data.body_lin_acc_w[:, asset_cfg.body_ids[0], :]

    if axis in ("x", "y"):
        base_acc_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), base_acc_w)
        return torch.square(base_acc_yaw[:, 0 if axis == "x" else 1])
    if axis == "z":
        return torch.square(base_acc_w[:, 2])
    raise ValueError(f"Unsupported acceleration axis {axis!r}; use 'x', 'y', or 'z'.")


def joint_torque_over_nominal(
    env: ManagerBasedRLEnv,
    nominal_torque: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize only applied torque above the nominal actuator torque."""

    if nominal_torque <= 0.0:
        raise ValueError("nominal_torque must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    applied_torque = torch.abs(robot.data.applied_torque[:, asset_cfg.joint_ids])
    return torch.sum(torch.clamp(applied_torque - nominal_torque, min=0.0), dim=1)


def leg_roll_velocity_excess(
    env: ManagerBasedRLEnv,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return normalized leg-roll speed above a dead-zone threshold."""

    if velocity_threshold <= 0.0:
        raise ValueError("velocity_threshold must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    joint_velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
    excess_velocity = torch.clamp(
        torch.abs(joint_velocity) - velocity_threshold, min=0.0
    )
    return torch.sum(excess_velocity / velocity_threshold, dim=1)


def leg_outward_roll_excess(
    env: ManagerBasedRLEnv,
    angle_threshold: float,
    outward_direction_signs: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return normalized outward leg-roll deviation beyond a dead zone."""

    if angle_threshold <= 0.0:
        raise ValueError("angle_threshold must be positive.")
    if len(outward_direction_signs) != 2:
        raise ValueError("outward_direction_signs must contain right and left signs.")

    robot: Articulation = env.scene[asset_cfg.name]
    joint_position = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_position = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    if joint_position.shape[1] != 2:
        raise ValueError("asset_cfg must resolve exactly two ordered leg-roll joints.")

    direction_signs = joint_position.new_tensor(outward_direction_signs)
    outward_deviation = (
        joint_position - default_joint_position
    ) * direction_signs
    excess_angle = torch.clamp(outward_deviation - angle_threshold, min=0.0)
    return torch.sum(excess_angle / angle_threshold, dim=1)


def is_any_terminated_term(
    env: ManagerBasedRLEnv, term_keys: str | list[str]
) -> torch.Tensor:
    """Return one when any selected non-timeout termination is active."""

    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in env.termination_manager.find_terms(term_keys):
        terminated |= env.termination_manager.get_term(term_name).bool()
    return terminated.float()


def contacting_feet_no_slide(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward every contacting foot for remaining still on the ground."""

    if std <= 0.0:
        raise ValueError("std must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_velocity_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    no_slide_score = torch.exp(
        -torch.sum(torch.square(foot_velocity_xy), dim=2) / std**2
    )
    contact_count = torch.sum(in_contact, dim=1)
    return torch.sum(no_slide_score * in_contact.float(), dim=1) / torch.clamp(
        contact_count, min=1
    )


def ground_contact_flatness_with_landing_bonus(
    env: ManagerBasedRLEnv,
    flat_tolerance: float,
    penalty_start_angle: float,
    landing_bonus: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Score both contacting feet and add emphasis on each touchdown frame."""

    if flat_tolerance < 0.0:
        raise ValueError("flat_tolerance must be non-negative.")
    if not flat_tolerance < penalty_start_angle < 0.5 * math.pi:
        raise ValueError("penalty_start_angle must be between tolerance and 90 degrees.")
    if landing_bonus < 0.0:
        raise ValueError("landing_bonus must be non-negative.")

    robot: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids]
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    if foot_quat_w.shape[1] != contact_time.shape[1]:
        raise ValueError("asset_cfg and sensor_cfg must resolve the same feet.")

    sole_normal_b = torch.zeros_like(foot_quat_w[..., 1:])
    sole_normal_b[..., 2] = 1.0
    sole_normal_w = quat_apply(
        foot_quat_w.reshape(-1, 4), sole_normal_b.reshape(-1, 3)
    ).reshape_as(sole_normal_b)
    tilt = torch.atan2(
        torch.linalg.vector_norm(sole_normal_w[..., :2], dim=-1),
        sole_normal_w[..., 2],
    )
    flat_score = (tilt <= flat_tolerance).float()
    tilt_penalty = torch.clamp(
        (tilt - penalty_start_angle) / (0.5 * math.pi - penalty_start_angle),
        min=0.0,
        max=1.0,
    )
    foot_score = flat_score - tilt_penalty
    in_contact = contact_time > 0.0
    contact_count = torch.sum(in_contact, dim=1)
    contact_score = torch.sum(foot_score * in_contact.float(), dim=1) / torch.clamp(
        contact_count, min=1
    )

    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    landing_count = torch.sum(first_contact, dim=1)
    landing_score = torch.sum(foot_score * first_contact.float(), dim=1) / torch.clamp(
        landing_count, min=1
    )
    landing_score *= (landing_count > 0).float()
    return contact_score + landing_bonus * landing_score


def time_off_ground_value(
    env: ManagerBasedRLEnv,
    lift_command: torch.Tensor,
    swing_contact: torch.Tensor,
    base_value: float,
    growth_rate: float,
) -> torch.Tensor:
    """Return a shared multiplier that grows while the commanded foot stays airborne."""

    if base_value < 0.0:
        raise ValueError("base_value must be non-negative.")
    if growth_rate < 0.0:
        raise ValueError("growth_rate must be non-negative.")

    state = getattr(env, _TIME_OFF_GROUND_STATE_ATTR, None)
    if state is None or state["air_time"].shape[0] != env.num_envs:
        state = {
            "air_time": torch.zeros(env.num_envs, device=env.device),
            "last_step": -1,
        }
        setattr(env, _TIME_OFF_GROUND_STATE_ATTR, state)

    current_step = int(env.common_step_counter)
    if state["last_step"] != current_step:
        airborne_during_lift = lift_command & ~swing_contact
        state["air_time"] = torch.where(
            airborne_during_lift,
            state["air_time"] + env.step_dt,
            torch.zeros_like(state["air_time"]),
        )
        state["last_step"] = current_step

    return base_value + growth_rate * state["air_time"]


def swing_foot_airborne(
    env: ManagerBasedRLEnv,
    time_off_ground_base_value: float,
    time_off_ground_growth_rate: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward commanded lifting and penalize airborne feet while standing."""

    command = env.command_manager.get_command(command_name)
    lift_command = command[:, 0] > 0.5
    support_index = command[:, 1].long()
    swing_index = 1 - support_index
    rows = torch.arange(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    support_contact = in_contact[rows, support_index]
    swing_contact = in_contact[rows, swing_index]
    air_time_value = time_off_ground_value(
        env,
        lift_command,
        swing_contact,
        time_off_ground_base_value,
        time_off_ground_growth_rate,
    )
    lift_score = (support_contact & ~swing_contact).float() * air_time_value
    # stand_penalty = -torch.any(~in_contact, dim=1).float()
    stand_penalty=0
    return torch.where(lift_command, lift_score, stand_penalty)


def swing_knee_flexion(
    env: ManagerBasedRLEnv,
    saturation_angle: float,
    correct_direction_signs: tuple[float, float],
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward additional human-like flexion of the selected swing knee."""

    if saturation_angle <= 0.0:
        raise ValueError("saturation_angle must be positive.")
    if len(correct_direction_signs) != 2:
        raise ValueError("correct_direction_signs must contain right and left signs.")

    robot: Articulation = env.scene[asset_cfg.name]
    knee_position = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_knee_position = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    if knee_position.shape[1] != 2:
        raise ValueError("asset_cfg must resolve exactly two ordered knee joints.")

    direction_signs = knee_position.new_tensor(correct_direction_signs)
    flexion_from_default = (knee_position - default_knee_position) * direction_signs

    command = env.command_manager.get_command(command_name)
    lift_command = command[:, 0] > 0.5
    support_index = command[:, 1].long()
    swing_index = 1 - support_index
    rows = torch.arange(env.num_envs, device=env.device)
    swing_flexion = flexion_from_default[rows, swing_index]

    score = torch.clamp(swing_flexion / saturation_angle, max=1.0)
    return score * lift_command.float()


class one_foot_command_reward(ManagerTermBase):
    """Reward the selected support/swing feet according to the binary command."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sole_vertices = cfg.params["sole_vertices"]
        if len(sole_vertices) != 2 or any(len(vertices) < 3 for vertices in sole_vertices):
            raise ValueError("sole_vertices must contain at least three vertices per foot.")
        self._sole_vertices = torch.tensor(
            sole_vertices, dtype=torch.float, device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_foot_lift_height: float,
        command_zero_weight: float,
        command_one_weight: float,
        time_off_ground_base_value: float,
        time_off_ground_growth_rate: float,
        sole_vertices,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del sole_vertices
        if max_foot_lift_height <= 0.0:
            raise ValueError("max_foot_lift_height must be positive.")

        robot: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids]
        foot_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids]
        num_envs, num_feet = foot_pos_w.shape[:2]
        if num_feet != 2 or self._sole_vertices.shape[0] != 2:
            raise ValueError("Exactly two ordered feet are required.")

        num_vertices = self._sole_vertices.shape[1]
        vertices = self._sole_vertices.unsqueeze(0).expand(num_envs, -1, -1, -1)
        quaternions = foot_quat_w.unsqueeze(2).expand(-1, -1, num_vertices, -1)
        rotated_vertices = quat_apply(
            quaternions.reshape(-1, 4), vertices.reshape(-1, 3)
        ).reshape(num_envs, num_feet, num_vertices, 3)
        sole_vertex_z_w = (
            foot_pos_w[:, :, 2].unsqueeze(2) + rotated_vertices[..., 2]
        )
        ground_z = env.scene.env_origins[:, 2].view(num_envs, 1, 1)
        sole_vertex_height = torch.clamp(sole_vertex_z_w - ground_z, min=0.0)
        minimum_sole_height = torch.amin(sole_vertex_height, dim=2)
        maximum_sole_height = torch.amax(sole_vertex_height, dim=2)

        command = env.command_manager.get_command(command_name)
        lift_command = command[:, 0] > 0.5
        support_index = command[:, 1].long()
        swing_index = 1 - support_index
        rows = torch.arange(env.num_envs, device=env.device)

        in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        support_contact = in_contact[rows, support_index]
        swing_contact = in_contact[rows, swing_index]
        swing_minimum_height = minimum_sole_height[rows, swing_index]
        swing_maximum_height = maximum_sole_height[rows, swing_index]

        lift_score = torch.clamp(
            swing_minimum_height / max_foot_lift_height, min=0.0, max=1.0
        )
        lift_score *= (support_contact & ~swing_contact).float()
        air_time_value = time_off_ground_value(
            env,
            lift_command,
            swing_contact,
            time_off_ground_base_value,
            time_off_ground_growth_rate,
        )

        lower_score = 1.0 - swing_maximum_height / max_foot_lift_height
        lower_score *= support_contact.float()
        return torch.where(
            lift_command,
            command_one_weight * lift_score * air_time_value,
            command_zero_weight * lower_score,
        )
