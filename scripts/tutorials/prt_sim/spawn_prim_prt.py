import argparse

from isaaclab.app import AppLauncher

parse = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")

AppLauncher.add_app_launcher_args(parse)

args_cli = parse.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
 
def design_scene():
	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
	cfg_light_distant = sim_utils.DistantLightCfg(
		intensity = 3000.0,
		color = (0.75, 0.75, 0.75),
	)
	cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
	prim_utils.create_prim("/World/Objects", "Xform")
	cfg_cone = sim_utils.ConeCfg(
		radius = 0.15,
		height = 0.5,
		visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
	)
	cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
	cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

	cfg_cone_rigid = sim_utils.ConeCfg(
		radius = 0.15,
		height = 0.5,
		rigid_props = sim_utils.RigidBodyPropertiesCfg(),
		mass_props = sim_utils.MassPropertiesCfg(mass=1.0),
		collision_props = sim_utils.CollisionPropertiesCfg(),
		visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
	)
	
	cfg_cone_rigid.func(
		"/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
	)
	cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
   
	cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

def main():
	"""Main function to run the scene design."""
	 # Initialize the simulation context
	sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
	sim = sim_utils.SimulationContext(sim_cfg)
	# Set main camera
	sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

	# Design scene by adding assets to it
	design_scene()

	# Play the simulator
	sim.reset()
	# Now we are ready!
	print("[INFO]: Setup complete...")

    # Simulate physics
	while simulation_app.is_running():
        # perform step
		sim.step()


if __name__ == "__main__":
	# Run the main function
	main()
	# Close the simulation app
	simulation_app.close()