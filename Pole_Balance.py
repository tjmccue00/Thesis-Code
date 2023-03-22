import Environment as en
import Asset as ast
import Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import matplotlib.pyplot as plt

model_file = r"/home/tjmccue/Documents/Thesis/Cart and Pendulum Model/URDF Model v8/Cart and Pendelum Assembly/urdf"
model_file_name = r"Cart and Pendelum Assembly.urdf"

gym = gymapi.acquire_gym()

sim = sm.Simulation(gym, dt=1/360, num_position_iterations=100, num_velocity_iterations=100)
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

stiffness = (0.0, 0.0)
damping = (0.0, 0.0)
friction = (0.0, 0.0)
max_velo = (1000, 1000)
max_effort = (800, 800)
env.set_actor_dof_props(stiffness, damping, friction, max_effort, max_velo)

joint_pole = gym.find_actor_dof_handle(env.envs[0], env.actors[0], "Pendelum_Rotational")
joint_cart = gym.find_actor_dof_handle(env.envs[0], env.actors[0], "Cart_Linear")

gym.apply_dof_effort(env.envs[0], joint_pole, 0)
gym.apply_dof_effort(env.envs[0], joint_cart, 0)

cam_pos = gymapi.Vec3(0, .25, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(sim.get_Camera(), None, cam_pos, cam_target)

env.asset.joint_pos[1] = 0.26

gym.set_actor_dof_states(env.envs[0], env.actors[0], asset.joint_state, gymapi.STATE_ALL)

sim_time = []
curr_time = 0
theta = []
counter = 0

while not gym.query_viewer_has_closed(sim.get_Camera()):

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)
    
    cart_pos = gym.get_joint_position(env.envs[0], joint_cart)
    pos = gym.get_joint_position(env.envs[0], joint_pole)
    velo = gym.get_joint_velocity(env.envs[0], joint_pole)
    velo_cart = gym.get_joint_velocity(env.envs[0], joint_cart)
    force = 250*pos+7.5*velo
    env.apply_force(joint_cart, force, 0)

    if curr_time < 0.75 and counter % 15 == 0:
        sim_time.append(curr_time)
        theta.append(pos)

    curr_time += sim.dt
    env.render()
    counter += 1

sim.end_sim()

with open("/home/tjmccue/Downloads/pend_response.csv", "r") as file:
    data = file.read()

data = data.split("\n")
time = []
theta_mat = []
for i in range(1, len(data)-1):
    temp = data[i].replace(" sec", "")
    temp = temp.split(",")
    time.append(float(temp[0]))
    theta_mat.append(float(temp[1]))
fig, ax = plt.subplots()
ax.plot(time, theta_mat, c="blue", label="MatLab Data")
ax.plot(sim_time, theta, "ro", label="Isaac Gym Data")
ax.set_xlabel("Time (t), sec")
ax.set_ylabel("Theta, rads")
plt.legend()
plt.show()