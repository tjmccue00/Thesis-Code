import math
import numpy as np
from isaacgym import gymapi, gymutil

class Environment():

    def __init__(self, asset, sim, gym, num_env, num_per_row, size_env):

        self.asset = asset
        self.sim = sim
        self.gym = gym
        self.num_env = num_env
        self.num_per_row = num_per_row
        self.size_env = size_env
        self.envs = []
        self.actors = []
        self.render = False
        self.sync_time = False

    
    def initialize(self):

        up_bound = gymapi.Vec3(self.size_env/2, self.size_env, self.size_env/2)
        low_bound = gymapi.Vec3(-self.size_env/2, 0, -self.size_env/2)

        for i in range(self.num_env):
            environ = self.gym.create_env(self.sim, low_bound, up_bound, self.num_per_row)
            self.envs.append(environ)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(self.asset.trans_t[0], self.asset.trans_t[1], self.asset.trans_t[2])
            pose.r = gymapi.Quat(self.asset.rotat_t[0], self.asset.rotat_t[0], self.asset.rotat_t[2], self.asset.rotat_t[3])

            self.actors.append(self.gym.create_actor(environ, self.asset, pose, self.asset.name + " " + str(i)))

    def render(self):

        if self.render:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.sim.camera, self.sim, True)
        
        if self.sync_time:
            self.gym.sync_frame_time(self.sim)
