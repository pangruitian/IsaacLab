import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets import CARTPOLE_CFG 
def design_scene():

	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
	cfg_dome_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
	cfg_dome_light.func("/World/Light", cfg_dome_light)
	origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
	prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
	prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
	cartpole_cfg = CARTPOLE_CFG.copy()
	cartpole_cfg.prim_path = "/World/Origin.*/Robot"
	cartpole = Articulation(cfg=cartpole_cfg)
	scene_entities = {"cartpole": cartpole}
	return scene_entities, origins

def run_simulation(sim: SimulationContext, scene_entities: dict, origins: torch.Tensor):
	sim_dt = sim.get_physics_dt()
	sim_time = 0.0
	count = 0
	robot = scene_entities["cartpole"]
	while simulation_app.is_running():
		if count % 500 == 0:
			count = 0
			root_state = robot.data.default_root_state.clone()
			root_state[:, :3] += origins
			robot.write_root_pose_to_sim(root_state[:, :7])
			robot.write_root_velocity_to_sim(root_state[:, 7:])
			joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
			joint_pos += torch.rand_like(joint_pos) * 0.1
			robot.write_joint_state_to_sim(joint_pos, joint_vel)
			robot.reset()
			print("[INFO]: Resetting robot state...")
		efforts = torch.randn_like(robot.data.joint_pos) * 10.0
		robot.set_joint_effort_target(efforts)
		robot.write_data_to_sim()
		sim.step()
		count += 1
		robot.update(sim_dt)

def main():
	sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
	sim = SimulationContext(cfg=sim_cfg)
	sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
	scene_entities, origins = design_scene()
	origins = torch.tensor(origins, device=args_cli.device)
	sim.reset()
	run_simulation(sim, scene_entities, origins)

if __name__ == "__main__":
	main()
	simulation_app.close()  # Close the application when done