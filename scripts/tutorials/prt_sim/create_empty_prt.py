# 终端参数的库
import argparse
from isaaclab.app import AppLauncher

# 创建解析器，parser是argparse.ArgumentParser类的实例，用于解析命令行参数
parser = argparse.ArgumentParser(description="创建一个空的 PRT 文件")
# 向parser中添加多个与定义的命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 把解析得到的参数给applauncher启动器，并创建启动器实例
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# 导入模拟配置和上下文

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
	#设置仿真时间步长
	sim_cfg = SimulationCfg(dt=0.01)
	#实例化一个 SimulationContext 对象，传入配置参数
	sim = SimulationContext(cfg=sim_cfg)
	sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
	#重置仿真环境并启动
	sim.reset()
	print("[INFO]: Setup complete...")
	#让仿真环境持续运行
	while simulation_app.is_running():
		#执行仿真步骤
		sim.step()

if __name__ == "__main__":
	#运行主函数
	main()
	#关闭仿真应用
	simulation_app.close()