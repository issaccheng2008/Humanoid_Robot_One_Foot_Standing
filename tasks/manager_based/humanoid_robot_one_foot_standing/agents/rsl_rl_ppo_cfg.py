# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


ENTROPY_COEF_CHANGE_AFTER_ITERATION = 1000
ENTROPY_COEF_AFTER_CHANGE = 0.001


@configclass
class ScheduledEntropyPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with a checkpoint-aware entropy coefficient change."""

    class_name: str = f"{__package__}.scheduled_entropy_ppo:ScheduledEntropyPPO"
    entropy_coef_change_after_iteration: int = ENTROPY_COEF_CHANGE_AFTER_ITERATION
    entropy_coef_after_change: float = ENTROPY_COEF_AFTER_CHANGE


@configclass
class HumanoidRobotOneFootStandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for canonical one-foot standing."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "humanoid_robot_one_foot_standing"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
    )

    algorithm = ScheduledEntropyPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
