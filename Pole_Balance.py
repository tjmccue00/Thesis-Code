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

dof_states = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_EFFORT)

env.set_actor_dof_states(dof_states)

joint_pole = gym.find_actor_dof_handle(env.envs[0], env.actors[0], "Rotational")
joint_cart = gym.find_actor_dof_handle(env.envs[0], env.actors[0], "Linear")

gym.apply_dof_effort(env.envs[0], joint_pole, 0)
#gym.apply_dof_effort(env.envs[0], joint_cart, 0)

cam_pos = gymapi.Vec3(0, .25, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(sim.get_Camera(), None, cam_pos, cam_target)
#target = 0
counter = 0
dat_rec = False
while not gym.query_viewer_has_closed(sim.get_Camera()):

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    if keyboard.is_pressed('w'):
        env.asset.joint_pos[1] -= 0.025
    elif keyboard.is_pressed('s'):
        env.asset.joint_pos[1] += 0.025
    elif keyboard.is_pressed('a'):
        env.asset.joint_pos[0] -= 0.025
    elif keyboard.is_pressed('d'):
        env.asset.joint_pos[0] += 0.025
    

    pos = gym.get_dof_position(env.envs[0], joint_pole)
    #print("Target: ",round(target,2) , "Actual: ", round(pos,2))
    print(round(pos,2))

    #env.asset.joint_pos[0] -= 0.125* pos
    gym.set_actor_dof_states(env.envs[0], env.actors[0], asset.joint_state, gymapi.STATE_ALL)
    #if dat_rec:
        #if counter % 10 == 0:
            #with open("pend_txt.txt", "a") as f:
                #f.write(str(round(target,2)) + "," + str(round(pos,2)) + "\n")

    #counter += 1

    env.render()

sim.end_sim()