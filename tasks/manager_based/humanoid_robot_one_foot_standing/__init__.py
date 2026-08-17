# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium environments for custom humanoid one-foot standing."""

import gymnasium as gym

from . import agents


gym.register(
    id="Humanoid-Robot-One-Foot-Standing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.humanoid_robot_one_foot_standing_env_cfg:"
            "HumanoidRobotOneFootStandingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "HumanoidRobotOneFootStandingPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Humanoid-Robot-One-Foot-Standing-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.humanoid_robot_one_foot_standing_env_cfg:"
            "HumanoidRobotOneFootStandingEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "HumanoidRobotOneFootStandingPPORunnerCfg"
        ),
    },
)
