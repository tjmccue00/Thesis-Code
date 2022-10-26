import math
import numpy as np
from isaacgym import gymapi, gymutil

def get_quat_from_eul(rpy):
    roll = rpy[0]*math.pi/180
    pitch = rpy[1]*math.pi/180
    yaw = rpy[2]*math.pi/180

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2)

    return gymapi.Quat(qw, qx, qy, qz)

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
        self.render_val = False
        self.sync_time = False

    
    def initialize(self):

        up_bound = gymapi.Vec3(self.size_env/2, self.size_env, self.size_env/2)
        low_bound = gymapi.Vec3(-self.size_env/2, 0, -self.size_env/2)

        for i in range(self.num_env):
            environ = self.gym.create_env(self.sim.sim, low_bound, up_bound, self.num_per_row)
            self.envs.append(environ)



            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(self.asset.trans_t[0], self.asset.trans_t[1], self.asset.trans_t[2])
            pose.r = get_quat_from_eul(self.asset.rotat_t)

            actor = self.gym.create_actor(environ, self.asset.asset, pose, self.asset.name + " " + str(i))
            self.actors.append(actor)

            self.gym.set_actor_dof_states(environ, actor, self.asset.joint_state, gymapi.STATE_ALL)

    def render(self):

        if self.render_val:
            self.gym.step_graphics(self.sim.sim)
            self.gym.draw_viewer(self.sim.camera, self.sim.sim, True)
        
        if self.sync_time:
            self.gym.sync_frame_time(self.sim.sim)


    def set_Render(self, val):
        self.render_val = val

    def set_Sync(self, val):
        self.sync_time = val