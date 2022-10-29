import Environment as en
import Asset as ast
import Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import keyboard

model_file = r"/home/tjmccue/Documents/Thesis/Cart and Pendulum Model/URDF Model v2/Cart and Pendelum Assembly/urdf"
model_file_name = r"Cart and Pendelum Assembly.urdf"

gym = gymapi.acquire_gym()

sim = sm.Simulation(gym)
sim.initialize()
sim.create_ground()

asset = ast.Asset(model_file, model_file_name, sim, gym, "Cart-Pole", rotat_t=[0, 0, -90])
asset.initialize()
asset.configure_joint_lims()

env = en.Environment(asset, sim, gym, 1, 1, 5)
env.initialize()

env.set_Render(True)
env.set_Sync(True)

while not gym.query_viewer_has_closed(sim.get_Camera()):

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    if keyboard.is_pressed('w'):
        env.asset.joint_pos[0] += 0.01
    elif keyboard.is_pressed('s'):
        env.asset.joint_pos[0] -= 0.01
    elif keyboard.is_pressed('a'):
        env.asset.joint_pos[1] += 0.05
    elif keyboard.is_pressed('d'):
        env.asset.joint_pos[1] -= 0.05

    gym.set_actor_dof_states(env.envs[0], env.actors[0], asset.joint_state, gymapi.STATE_ALL)

    env.render()

sim.end_sim()