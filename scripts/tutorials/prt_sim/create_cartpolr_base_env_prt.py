import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="基于管理器的Cartpole环境创建示例。")
parser.add_argument("--num_envs", type=int, default=16, help="要生成的环境数量。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余部分保持不变。"""
import math
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg

@configclass
class ActionsCfg:
	joint_efforts = mdp.JointEffortActionCfg(
		asset_name="robot", 
		joint_names=["slider_to_cart"], 
		scale=5.0)


@configclass
class ObservationsCfg:
	"""定义了一个观测组叫policy，包含两个观测项：joint_pos_rel和joint_vel_rel。"""
	@configclass
	class PolicyCfg(ObsGroup):
		joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel) #tensor数据
		joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel) #tensor数据

		def __post_init__(self) -> None:
			self.enable_corruption = False #不启动观察噪声
			self.concatenate_terms = True  #将所有观察项连接成一个向量

	policy: PolicyCfg = PolicyCfg()

@configclass
class EventsCfg:
	add_pole_mass = EventTerm(
		func=mdp.randomize_rigid_body_mass,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
			"position_range": (0, 0, 0),
			"velocity_range": (0, 0, 0),
		}
	)

	reset_cart_position = EventTerm(
		func=mdp.reset_joints_by_offset,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
			"position_range": (-1.0, 1.0),
			"velocity_range": (-0.1, 0.1),
		}
	)
	reset_pole_position = EventTerm(
		func=mdp.reset_joints_by_offset,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
			"position_range": (-0.125 * math.pi, 0.125 * math.pi),
			"velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
		},
	)

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
	