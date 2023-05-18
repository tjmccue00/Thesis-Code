import SimModules.Environment as en
import SimModules.Asset as ast
import SimModules.Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import random as rng
import RL.QLearning as ql
import matplotlib.pyplot as plt

model_file = r"/home/tjmccue/Documents/Thesis/Cart and Pendulum Model/URDF Model v8/Cart and Pendelum Assembly/urdf"
model_file_name = r"Cart and Pendelum Assembly.urdf"

gym = gymapi.acquire_gym()

sim = sm.Simulation(gym, dt=1/360)
sim.initialize()
sim.create_ground()

asset = ast.Asset(model_file, model_file_name, sim, gym, "Cart-Pole", rotat_t=[0, 0, 180])
asset.initialize()
asset.configure_joint_lims()

env = en.Environment(asset, sim, gym, 1, 2, 2)
env.initialize()

env.set_Render(False)
env.set_Sync(False)

rng.seed(3289457)
for i in range(len(env.envs)):

    env.asset.joint_pos[1] = rng.randint(-2, 2)/10

    gym.set_actor_dof_states(env.envs[i], env.actors[i], asset.joint_state, gymapi.STATE_ALL)

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
friction = (0.0, 0.00)
max_velo = (1000, 1000)
max_effort = (800, 800)
env.set_actor_dof_props(stiffness, damping, friction, max_effort, max_velo)

cam_pos = gymapi.Vec3(0, .25, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(sim.get_Camera(), None, cam_pos, cam_target)

qlearn = ql.Agent(10, 4, [-200, 200], [[-.2095, .2095], [-10, 10], [-0.62, 0.62], [-10, 10]], 0.2, 0.95, 30)
qlearn.initialize_table()
epsilon = 0.25
episodes = 5000
done = False
rewards = 0
scores = [0]*len(env.envs)
current_states = [(0,0,0,0)]*len(env.envs)
current_actions = [0]*len(env.envs)

for i in range(len(env.envs)):

    cart_pos = 0
    pos = gym.get_joint_position(env.envs[i], pole_joints[i])
    velo = 0
    velo_cart = 0
    current_states[i] = qlearn.discretize((pos, velo, cart_pos, velo_cart))

episode = 0

env.set_Render(False)
env.set_Sync(False)
all_scores = []
avg_scores = []
all_episodes = []
timestep = [0]*len(env.envs)

while episode < episodes:

    
    score = [0] * len(env.envs)

    for i in range(len(env.envs)):


        if np.random.uniform(0,1) < epsilon:
            
            action, action_idx = qlearn.get_sample_action()
        
        else:

            action, action_idx = qlearn.get_action(qlearn.discretize((pos, velo, cart_pos, velo_cart)))


        env.apply_force(cart_joints[i], action, i)

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    for i in range(len(env.envs)):
        
        

        cart_pos = gym.get_joint_position(env.envs[i], cart_joints[i])
        pos = gym.get_joint_position(env.envs[i], pole_joints[i])
        velo = gym.get_joint_velocity(env.envs[i], pole_joints[i])
        velo_cart = gym.get_joint_velocity(env.envs[i], cart_joints[i])

        next_state = qlearn.discretize((pos, velo, cart_pos, velo_cart))

        if (abs(cart_pos) > 0.62) or (abs(pos) > .2095):
            done = True
            episode += 1

        
        if not done:
            reward = 1
            scores[i] += reward
            
            current_states[i] = next_state
            timestep[i] += sim.dt
            qlearn.update_table(next_state, current_states[i], action_idx, reward)

        else:
            env.asset.joint_pos[1] = rng.randint(-2, 2)/10
            env.asset.joint_pos[0] = 0
            

            gym.set_actor_dof_states(env.envs[i], env.actors[i], asset.joint_state, gymapi.STATE_ALL)

            cart_pos = 0
            pos = gym.get_joint_position(env.envs[i], pole_joints[i])
            velo = 0
            velo_cart = 0

            current_states[i] = qlearn.discretize((pos, velo, cart_pos, velo_cart))
            rewards += scores[i]
            all_scores.append(scores[i])
            avg_score = np.mean(all_scores[-100:])
            avg_scores.append(avg_score)
            all_episodes.append(episode)
            if episode % 10 == 0:
                print("Episode: ", episode, "Rewards: ", round(rewards,1), "Average Score: ", round(avg_score), "Episode Time: ", round(timestep[i],1), epsilon)
            scores[i] = 0
            timestep[i] = 0
            current_actions[i] = 0
            done = False

        
    env.render()



sim.end_sim()

avg_scores = avg_scores[0:episodes]
all_episodes = all_episodes[0:episodes]

fig, ax = plt.subplots()
ax.scatter(all_episodes, avg_scores, c="green")
ax.set_xlabel("Episode")
ax.set_ylabel("Avg. Score")
plt.show()
