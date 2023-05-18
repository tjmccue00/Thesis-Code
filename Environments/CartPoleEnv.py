import Environments.SimModules.Environment as en
import Environments.SimModules.Asset as ast
import Environments.SimModules.Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import random as rng


class CartPole(object):

    def __init__(self):
        
        self.model_file = r"/home/tjmccue/Documents/Thesis/Cart and Pendulum Model/URDF Model v8/Cart and Pendelum Assembly/urdf"
        self.model_file_name = r"Cart and Pendelum Assembly.urdf"

        self.gym = gymapi.acquire_gym()

        self.sim = sm.Simulation(self.gym, dt=1/360)
        self.sim.initialize()
        self.sim.create_ground()

        self.asset = ast.Asset(self.model_file, self.model_file_name, self.sim, self.gym, "Cart-Pole", rotat_t=[0, 0, 180])
        self.asset.initialize()
        self.asset.configure_joint_lims()

        self.env = en.Environment(self.asset, self.sim, self.gym, 1, 2, 2)
        self.env.initialize()

        cam_pos = gymapi.Vec3(0, .25, 2)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.sim.get_Camera(), None, cam_pos, cam_target)

        self.env.set_Render(True)
        self.env.set_Sync(True)

        self.dof_states = (gymapi.DOF_MODE_EFFORT, gymapi.DOF_MODE_NONE)

        self.env.set_actor_dof_states(self.dof_states)

        self.pole_joint = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Pendelum_Rotational")
        self.cart_joint = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Cart_Linear")

        stiffness = (0.0, 0.0)
        damping = (0.0, 0.0)
        friction = (0.0, 0.00)
        max_velo = (1000, 1000)
        max_effort = (800, 800)
        self.env.set_actor_dof_props(stiffness, damping, friction, max_effort, max_velo)

        self.observation_space = 4



    def reset(self):
            
            self.env.asset.joint_pos[1] = rng.randint(-2, 2)/10

            self.gym.set_actor_dof_states(self.env.envs[0], self.env.actors[0], self.asset.joint_state, gymapi.STATE_ALL)

            cart_pos = 0
            pos = self.gym.get_joint_position(self.env.envs[0], self.pole_joint)
            velo = 0
            velo_cart = 0
            observation = (pos, velo, cart_pos, velo_cart)

            return observation


    def step(self, action):

        self.env.apply_force(self.cart_joint, action, 0)
         
        self.gym.simulate(self.sim.sim)
        self.gym.fetch_results(self.sim.sim, True)

        cart_pos = self.gym.get_joint_position(self.env.envs[0], self.cart_joint)
        pos = self.gym.get_joint_position(self.env.envs[0], self.pole_joint)
        velo = self.gym.get_joint_velocity(self.env.envs[0], self.pole_joint)
        velo_cart = self.gym.get_joint_velocity(self.env.envs[0], self.cart_joint)

        observation = (pos, velo, cart_pos, velo_cart)

        if (abs(cart_pos) > 0.62) or (abs(pos) > .2095):
             done = True
             reward = 1
        else:
             done = False
             reward = 1

        return observation, reward, done
    
    def render(self):
         
         self.env.render()

    def close(self):
         
         self.sim.end_sim()
