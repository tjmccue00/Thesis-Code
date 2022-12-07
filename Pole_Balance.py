import Environment as en
import Asset as ast
import Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import keyboard

model_file = r"/home/tjmccue/Documents/Thesis/Cart and Pendulum Model/URDF Model v8/Cart and Pendelum Assembly/urdf"
model_file_name = r"Cart and Pendelum Assembly.urdf"

gym = gymapi.acquire_gym()

sim = sm.Simulation(gym, dt=1/120)
sim.initialize()
sim.create_ground()

asset = ast.Asset(model_file, model_file_name, sim, gym, "Cart-Pole", rotat_t=[0, 0, 180])
asset.initialize()
asset.configure_joint_lims()

env = en.Environment(asset, sim, gym, 1, 1, 5)
env.initialize()

env.set_Render(True)
env.set_Sync(True)

dof_states = (gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_NONE)

env.set_actor_dof_states(dof_states)

joint_pole = gym.find_actor_dof_handle(env.envs[0], env.actors[0], "Pendelum_Rotational")
joint_cart = gym.find_actor_dof_handle(env.envs[0], env.actors[0], "Cart_Linear")

gym.apply_dof_effort(env.envs[0], joint_pole, 0)
gym.apply_dof_effort(env.envs[0], joint_cart, 0)

cam_pos = gymapi.Vec3(0, .25, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(sim.get_Camera(), None, cam_pos, cam_target)

counter = 0
dat_rec = True

while not gym.query_viewer_has_closed(sim.get_Camera()):

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    if keyboard.is_pressed('w'):
        #env.apply_force(joint_pole, 50)
        pass
    elif keyboard.is_pressed('s'):
        #env.apply_force(joint_pole, -50)
        pass
    elif keyboard.is_pressed('a'):
        env.apply_force(joint_cart, -50)
        force_u = 1
    elif keyboard.is_pressed('d'):
        env.apply_force(joint_cart, 50)
        force_u = 1
    else:
        force_u = 0
    
    cart_pos = gym.get_joint_position(env.envs[0], joint_cart)
    pos = gym.get_joint_position(env.envs[0], joint_pole)
    velo = gym.get_joint_velocity(env.envs[0], joint_pole)
    velo_cart = gym.get_joint_velocity(env.envs[0], joint_cart)
    force = 250*pos+7.5*velo
    #env.apply_force(joint_cart, force)
    
    print("Cart Pos: ", round(pos,2))
    print("Cart Velo: ", round(velo_cart,2))
    print("Pole Velo: ", round(velo,2))
    print("Force Applied: ", round(force,2))

    data = str(round(pos,2)) + ", " + str(round(velo_cart,2)) + ", " + str(round(velo,2)) + ", " + str(round(force,2)) + ", " + str(force_u)
    
    if dat_rec:
        if counter % 1 == 0:
            with open("pole_balance_no_control.txt", "a") as f:
                f.write(data + "\n")
        if counter == 120:
            env.apply_force(joint_cart, 200)
            pass

    counter += 1

    env.render()

sim.end_sim()