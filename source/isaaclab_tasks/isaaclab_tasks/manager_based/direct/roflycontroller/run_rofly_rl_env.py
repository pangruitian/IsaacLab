# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab_tasks.direct.roflycontroller.roflycontroller_env_cfg import RoflycontrollerEnvCfg
from isaaclab_tasks.direct.roflycontroller.roflycontroller_env import RoflycontrollerEnv

from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnvCfg
from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnv


def main():
    """Main function."""
    # create environment configuration

    # env_cfg = RoflycontrollerEnvCfg()
    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)
    # env = RoflycontrollerEnv(cfg=env_cfg)
    env = QuadcopterEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                # print("-" * 80)
                # print("[INFO]: Resetting environment...")
            # sample random actions
            ZERO_THRUST = -1.0
            a = torch.zeros(args_cli.num_envs, 4)
  
            # joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            # print(a)
            obs, rew, terminated, truncated, info = env.step(a)
            
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
