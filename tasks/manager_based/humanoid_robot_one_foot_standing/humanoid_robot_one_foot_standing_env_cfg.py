# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based task for stationary, symmetric one-foot standing."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .humanoid_robot import HUMANOID_ROBOT_CFG


SMALL_RANDOM_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.001,
    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,
    sub_terrains={
        "small_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.005, 0.005),
            noise_step=0.001,
            downsampled_scale=0.10,
            border_width=0.25,
        ),
    },
)


LEG_JOINT_NAMES = [
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]

# Preserve this order everywhere: index 0 is physical right, index 1 is left.
FOOT_BODY_NAMES = ["r_ankle_roll_link", "l_ankle_roll_link"]
BASE_BODY_NAME = "base_link"
ANKLE_JOINT_NAMES = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]

MIN_BASE_HEIGHT = 0.20
MAX_BASE_TILT = math.radians(65.0)
MAX_FOOT_LIFT_HEIGHT = 0.05
EL05_RATED_TORQUE = 1.5
ONE_FOOT_COMMAND_NAME = "lift_one_foot_in_the_air"

# Convex perimeters of the lowest physical sole surfaces, ordered right/left.
FOOT_SOLE_VERTICES = (
    (
        (0.045730848, 0.038124181, -0.043790001),
        (-0.104268424, 0.037654478, -0.043790001),
        (-0.108092859, 0.036881294, -0.043790001),
        (-0.111330278, 0.034703419, -0.043790001),
        (-0.113487840, 0.031452414, -0.043790001),
        (-0.114237063, 0.027623216, -0.043790001),
        (-0.114061706, -0.028376512, -0.043790001),
        (-0.113288522, -0.032200944, -0.043790001),
        (-0.111110643, -0.035438374, -0.043790001),
        (-0.107859641, -0.037595931, -0.043790001),
        (-0.104030438, -0.038345147, -0.043790001),
        (0.045968831, -0.037875444, -0.043790001),
        (0.049793262, -0.037102260, -0.043790001),
        (0.053030688, -0.034924384, -0.043790001),
        (0.055188250, -0.031673379, -0.043790001),
        (0.055937465, -0.027844181, -0.043790001),
        (0.055762108, 0.028155547, -0.043790001),
        (0.054988924, 0.031979978, -0.043790001),
        (0.052811045, 0.035217408, -0.043790001),
        (0.049560048, 0.037374966, -0.043790001),
    ),
    (
        (-0.113911822, 0.028337635, -0.043790001),
        (-0.114087179, -0.027662093, -0.043790001),
        (-0.113337964, -0.031491291, -0.043790001),
        (-0.111180402, -0.034742296, -0.043790001),
        (-0.107942976, -0.036920171, -0.043790001),
        (-0.104118548, -0.037693355, -0.043790001),
        (0.045880727, -0.038163058, -0.043790001),
        (0.049709924, -0.037413843, -0.043790001),
        (0.052960925, -0.035256285, -0.043790001),
        (0.055138804, -0.032018855, -0.043790001),
        (0.055911988, -0.028194424, -0.043790001),
        (0.056087345, 0.027805304, -0.043790001),
        (0.055338129, 0.031634502, -0.043790001),
        (0.053180564, 0.034885507, -0.043790001),
        (0.049943142, 0.037063383, -0.043790001),
        (0.046118710, 0.037836567, -0.043790001),
        (-0.103880562, 0.038306270, -0.043790001),
        (-0.107709758, 0.037557054, -0.043790001),
        (-0.110960759, 0.035399497, -0.043790001),
        (-0.113138638, 0.032162067, -0.043790001),
    ),
)


def _ordered_feet_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True)


def _ordered_feet_sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True
    )


@configclass
class HumanoidRobotOneFootStandingSceneCfg(InteractiveSceneCfg):
    """Randomly rough humanoid scene with IMU and foot-contact sensing."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SMALL_RANDOM_ROUGH_TERRAIN_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = HUMANOID_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        debug_vis=False,
        gravity_bias=(0.0, 0.0, 9.81),
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
        ),
    )


@configclass
class CommandsCfg:
    """Binary lift command; support side is randomized once per episode."""

    lift_one_foot_in_the_air = mdp.OneFootStandingCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        initial_stand_time_range_s=(1.0, 2.0),
        lift_time_range_s=(3.0, 5.0),
        support_left_probability=0.5,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Canonical joint-position actions mapped back to the physical side."""

    joint_pos = mdp.CanonicalJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
        command_name=ONE_FOOT_COMMAND_NAME,
    )


@configclass
class ObservationsCfg:
    """Policy observations canonicalized as right support and left swing."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_acc = ObsTerm(
            func=mdp.canonical_imu_lin_acc,
            params={
                "command_name": ONE_FOOT_COMMAND_NAME,
                "asset_cfg": SceneEntityCfg("imu"),
            },
            noise=Unoise(n_min=-0.3, n_max=0.3),
            scale=0.1,
        )
        base_ang_vel = ObsTerm(
            func=mdp.canonical_imu_ang_vel,
            params={
                "command_name": ONE_FOOT_COMMAND_NAME,
                "asset_cfg": SceneEntityCfg("imu"),
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.canonical_imu_projected_gravity,
            params={
                "command_name": ONE_FOOT_COMMAND_NAME,
                "asset_cfg": SceneEntityCfg("imu"),
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        lift_one_foot_in_the_air = ObsTerm(
            func=mdp.lift_one_foot_in_the_air,
            params={"command_name": ONE_FOOT_COMMAND_NAME},
        )
        joint_pos = ObsTerm(
            func=mdp.canonical_joint_pos_rel,
            params={
                "command_name": ONE_FOOT_COMMAND_NAME,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.canonical_joint_vel_rel,
            params={
                "command_name": ONE_FOOT_COMMAND_NAME,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(
            func=mdp.canonical_last_action,
            params={"action_name": "joint_pos"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Conservative resets and the reference project's domain randomization."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_foot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            ),
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
            ),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "add",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    """One-foot-standing objective and requested regularization terms."""

    base_lin_vel_zero = RewTerm(
        func=mdp.base_linear_velocity_zero,
        weight=2.0,
        params={"std": 0.15, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_ang_vel_zero = RewTerm(
        func=mdp.base_angular_velocity_zero,
        weight=3,
        params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot")},
    )
    contacting_feet_no_slide = RewTerm(
        func=mdp.contacting_feet_no_slide,
        weight=1.0,
        params={
            "std": 0.05,
            "asset_cfg": _ordered_feet_cfg(),
            "sensor_cfg": _ordered_feet_sensor_cfg(),
        },
    )
    ground_contact_flatness = RewTerm(
        func=mdp.ground_contact_flatness_with_landing_bonus,
        weight=0.5,
        params={
            "flat_tolerance": math.radians(5.0),
            "penalty_start_angle": math.radians(10.0),
            "landing_bonus": 1.0,
            "asset_cfg": _ordered_feet_cfg(),
            "sensor_cfg": _ordered_feet_sensor_cfg(),
        },
    )
    termination_penalty = RewTerm(
        func=mdp.is_any_terminated_term,
        weight=-200.0,
        params={"term_keys": ["bad_orientation", "low_base_height"]},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    base_acc_y_l2 = RewTerm(
        func=mdp.base_acceleration_l2,
        weight=-0.005,
        params={
            "axis": "y",
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
        },
    )
    base_acc_z_l2 = RewTerm(
        func=mdp.base_acceleration_l2,
        weight=-0.005,
        params={
            "axis": "z",
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
        },
    )
    base_acc_x_l2 = RewTerm(
        func=mdp.base_acceleration_l2,
        weight=-0.005,
        params={
            "axis": "x",
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
        },
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        },
    )
    dof_torque_over_nominal = RewTerm(
        func=mdp.joint_torque_over_nominal,
        weight=-0.1,
        params={
            "nominal_torque": EL05_RATED_TORQUE,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)
        },
    )
    one_foot_command = RewTerm(
        func=mdp.one_foot_command_reward,
        weight=3.0,
        params={
            "max_foot_lift_height": MAX_FOOT_LIFT_HEIGHT,
            "sole_vertices": FOOT_SOLE_VERTICES,
            "command_name": ONE_FOOT_COMMAND_NAME,
            "asset_cfg": _ordered_feet_cfg(),
            "sensor_cfg": _ordered_feet_sensor_cfg(),
        },
    )


@configclass
class TerminationsCfg:
    """Terminate only for timeout, excessive tilt, or a fallen base."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": MAX_BASE_TILT,
        },
    )
    low_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
        },
    )


@configclass
class HumanoidRobotOneFootStandingEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for a single canonical one-foot policy."""

    scene: HumanoidRobotOneFootStandingSceneCfg = HumanoidRobotOneFootStandingSceneCfg(
        num_envs=512, env_spacing=2.5
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)


@configclass
class HumanoidRobotOneFootStandingEnvCfg_PLAY(HumanoidRobotOneFootStandingEnvCfg):
    """Single-environment visualization configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.0, 2.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)
