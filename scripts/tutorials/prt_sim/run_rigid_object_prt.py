import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

def design_scene():
	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
	cfg_dome_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
	cfg_dome_light.func("/World/Light", cfg_dome_light)
	origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
	for i, origin in enumerate(origins):
		prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
	
	cone_cfg = RigidObjectCfg(
		prim_path="/World/Origin.*/Cone",
		spawn=sim_utils.ConeCfg(
			radius=0.1,
			height=0.2,
			rigid_props=sim_utils.RigidBodyPropertiesCfg(),
			mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(),

	)
	cone_object = RigidObject(cfg=cone_cfg)
	scene_entity = {"cone":cone_object}
	return scene_entity, origins

def run_simulation(sim:sim_utils.SimulationContext,entity:dict[str,RigidObject], origins:torch.Tensor):
	"""Run the simulation."""	
	sim_dt=sim.get_physics_dt()
	sim_time=0.0
	count=0
	cone_object = entity["cone"]
	while simulation_app.is_running():
		if count % 1000 == 0:
			sim_time = 0.0
			count=0
			root_state=cone_object.data.default_root_state.clone()
			root_state[:, :3] += origins
			root_state[:, :3] += math_utils.sample_cylinder(
				radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
			)
			cone_object.write_root_pose_to_sim(root_state[:, :7])
			cone_object.write_root_velocity_to_sim(root_state[:, 7:])
			cone_object.reset()
			print(">>>>>>>> Reset!")
			print(f"[INFO]: Resetting cone object at {root_state[:, :3]}")
		# cone_object.write_root_velocity_to_sim()
		sim.step()
		sim_time += sim_dt
		count += 1
		cone_object.update(sim_dt)
		if count % 100 == 0:
			print(f"[INFO]: Simulation time: {sim_time:.2f} seconds, Count: {count}")

def main():
	sim_cfg= sim_utils.SimulationCfg(device=args_cli.device)
	sim = SimulationContext(cfg=sim_cfg)

	sim.set_camera_view(eye=[0.0, 0.0, 1.0], target=[0.0, 0.0, 0.0])

	scene_entity, origins = design_scene()
	scene_origins = torch.tensor(origins, device=sim.device)
	sim.reset()
	print("[INFO]: Setup complete...")
	run_simulation(sim, scene_entity, scene_origins)

if __name__ == "__main__":
	main()
	simulation_app.close()
	print("[INFO]: Simulation finished and application closed.")