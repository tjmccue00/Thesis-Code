import Environment as en
import Asset as ast
import Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import random as rng

model_file = r"/home/tjmccue/Documents/Thesis/Cart and Pendulum Model/URDF Model v8/Cart and Pendelum Assembly/urdf"
model_file_name = r"Cart and Pendelum Assembly.urdf"

gym = gymapi.acquire_gym()

sim = sm.Simulation(gym, dt=1/60)
sim.initialize()
sim.create_ground()

asset = ast.Asset(model_file, model_file_name, sim, gym, "Cart-Pole", rotat_t=[0, 0, 180])
asset.initialize()
asset.configure_joint_lims()

env = en.Environment(asset, sim, gym, 4, 2, 2)
env.initialize()

env.set_Render(True)
env.set_Sync(True)

dof_states = (gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_NONE)

env.set_actor_dof_states(dof_states)

cart_joints = []
pole_joints = []

for i in range(len(env.envs)):

    joint_pole = gym.find_actor_dof_handle(env.envs[i], env.actors[i], "Pendelum_Rotational")
    joint_cart = gym.find_actor_dof_handle(env.envs[i], env.actors[i], "Cart_Linear")

    cart_joints.append(joint_cart)
    pole_joints.append(joint_pole)

stiffness = (0.0, 0.0)
damping = (0.0, 0.0)
friction = (0.1, 0.0025)
max_velo = (1000, 1000)
max_effort = (800, 800)
env.set_actor_dof_props(stiffness, damping, friction, max_effort, max_velo)

cam_pos = gymapi.Vec3(0, .25, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(sim.get_Camera(), None, cam_pos, cam_target)
rng.seed(3289457)

counters = [1, 61, 121, 181]

while not gym.query_viewer_has_closed(sim.get_Camera()):

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    for i in range(len(env.envs)):


        if counters[i] % 120 == 0:

            apply = rng.randrange(-200, 200, 5)
            env.apply_force(cart_joints[i], apply, i)
        
        counters[i] += 1

        cart_pos = gym.get_joint_position(env.envs[i], cart_joints[i])
        pos = gym.get_joint_position(env.envs[i], pole_joints[i])
        velo = gym.get_joint_velocity(env.envs[i], pole_joints[i])
        velo_cart = gym.get_joint_velocity(env.envs[i], cart_joints[i])
        force = 250*pos+7.5*velo
        env.apply_force(cart_joints[i], force, i)

    env.render()



sim.end_sim()