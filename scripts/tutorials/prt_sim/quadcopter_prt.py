import argparse
import math
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="基于管理器的四旋翼环境创建示例。")
parser.add_argument("--num_envs", type=int, default=16, help="要生成的环境数量。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余部分保持不变。"""
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG

# 场景配置
@configclass
class QuadcopterSceneCfg:
    """四旋翼场景配置"""
    # 地面
    terrain = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # 四旋翼机器人
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

# 动作配置
@configclass
class ActionsCfg:
    """四旋翼动作配置：使用外部力控制"""
    
    # 方法1：外部力/扭矩动作
    propeller_forces = mdp.ApplyExternalForceTorqueActionCfg(
        asset_name="robot",
        body_names=["m.*_prop"],  # 匹配所有螺旋桨
        scale=[[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]  # 只在Z方向施加力
    )
    # 动作维度：[num_envs, 4] - 每个螺旋桨一个推力
    
    # 方法2：关节力矩动作（如果有螺旋桨关节）
    # joint_efforts = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=["m.*_prop_joint"],  # 螺旋桨关节
    #     scale=100.0
    # )

# 观测配置
@configclass
class ObservationsCfg:
    """四旋翼观测配置"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        # 机体位置和姿态
        root_pos = ObsTerm(func=mdp.root_pos_w)           # 世界坐标系位置
        root_quat = ObsTerm(func=mdp.root_quat_w)         # 世界坐标系四元数
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_b)   # 机体坐标系线速度
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_b)   # 机体坐标系角速度
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

# 事件配置
@configclass
class EventsCfg:
    """四旋翼事件配置"""
    
    # 启动时随机化质量
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "mass_distribution_params": (0.8, 1.2),  # 质量范围：80%-120%
            "operation": "scale",
        },
    )
    
    # 重置时随机化初始位置
    reset_root_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0), 
                "z": (0.5, 2.0),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

# 完整环境配置
@configclass
class QuadcopterEnvCfg(ManagerBasedEnvCfg):
    """四旋翼环境配置"""
    
    # 场景设置
    scene: QuadcopterSceneCfg = QuadcopterSceneCfg(num_envs=1024, env_spacing=3.0)
    
    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    
    def __post_init__(self):
        """后初始化设置"""
        # 查看器设置
        self.viewer.eye = [5.0, 5.0, 3.0]
        self.viewer.lookat = [0.0, 0.0, 1.0]

def main():
    """主函数"""
    # 创建环境配置
    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 创建环境
    env = ManagerBasedEnv(cfg=env_cfg)
    
    print("[INFO]: 四旋翼环境创建完成...")
    print(f"[INFO]: 观测空间: {env.observation_manager.group_obs_dim}")
    print(f"[INFO]: 动作空间: {env.action_manager.total_action_dim}")
    
    # 获取悬停所需的推力（用于基线控制）
    robot = env.scene["robot"]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.device).norm()
    hover_thrust = robot_mass * gravity / 4.0  # 每个螺旋桨的悬停推力
    
    # 仿真循环
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 重置环境
            if count % 500 == 0:
                obs, _ = env.reset()
                print(f"[INFO]: 重置环境，观测维度: {obs['policy'].shape}")
            
            # 生成动作
            if count < 100:
                # 前100步：悬停控制
                actions = torch.full((env.num_envs, 4), hover_thrust / 10.0, device=env.device)
            else:
                # 之后：随机控制
                actions = torch.randn(env.num_envs, 4, device=env.device) * 0.5 + hover_thrust / 10.0
            
            # 执行动作
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # 打印信息
            if count % 100 == 0:
                avg_height = obs["policy"][:, 2].mean().item()  # Z位置
                print(f"步数: {count}, 平均高度: {avg_height:.3f}m")
        
        count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()