import math
import numpy as np
from isaacgym import gymapi, gymutil

def get_quat_from_eul(rpy):
    """
    Converts rpy coordinates to quaternion coords

    Input: Takes list of rpy coords in degrees
    Return: List of quaternion coords
    """
    roll = rpy[0]*math.pi/180
    pitch = rpy[1]*math.pi/180
    yaw = rpy[2]*math.pi/180

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2)

    return gymapi.Quat(qw, qx, qy, qz)

class Environment():
    """
    Holds all environment information as well as all the actors within, controls render settings and
    sets initial state of actors within environments
    """


    def __init__(self, asset, sim, gym, num_env, num_per_row, size_env):

        self.asset = asset
        self.sim = sim
        self.gym = gym
        self.num_env = num_env
        self.num_per_row = num_per_row
        self.size_env = size_env
        self.envs = []
        self.actors = []
        self.actor_props = []
        self.render_val = False
        self.sync_time = False

    
    def initialize(self):
        """
        Initializes environment object and sets up actors initial state within each environment created
        """


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

            self.actor_props.append(self.asset.joint_props)

    def render(self):
        """
        Renders and syncs frames to real time depending on attribute values
        """

        if self.render_val:
            self.gym.step_graphics(self.sim.sim)
            self.gym.draw_viewer(self.sim.camera, self.sim.sim, True)
        
        if self.sync_time:
            self.gym.sync_frame_time(self.sim.sim)


    def set_Render(self, val):
        """
        Sets whether to render the environment or not

        Input: Boolean value that is true to render, false otherwise
        """
        self.render_val = val

    def set_Sync(self, val):
        """
        Sets whether to sync frames to real time in the environment

        Input: Boolean value that is true to sync, false otherwise
        """
        self.sync_time = val

    def set_actor_dof_states(self, dof_modes):
        """
        Sets actor states based on list of DOF modes

        Input: List of gymapi Degree of Freedom modes that is len(joints)
        """
        self.asset.joint_props = dof_modes
        for i in range(len(self.envs)):
            self.actor_props[i]["driveMode"] = dof_modes
            self.gym.set_actor_dof_properties(self.envs[i], self.actors[i], self.actor_props[i])

    def set_actor_dof_props(self, stiffness, damping, friction, effort_max, velo_max):
        """
        Sets actor props based upon entered values

        Input:
            stiffness: List of stiffness values for each joint
            damping: List of damping values for each joint
            friction: List of friction values for each joint
            effort_max: List of max effort that can be applied to each joint
            velo_max: List of max velocity each joint can reach
        """

        for i in range(len(self.envs)):
            self.actor_props[i]["stiffness"] = (0.0, 0.0)
            self.actor_props[i]["damping"] = (0.0, 0.0)
            self.actor_props[i]["friction"] = (0.1, 0.0025)
            self.actor_props[i]["effort"] = (1000, 1000)
            self.actor_props[i]["velocity"] = (800, 800)
            self.gym.set_actor_dof_properties(self.envs[i], self.actors[i], self.actor_props[i])

    def apply_force(self, dof, force, environment):
        """
        Applies force to a specific joint within an environment

        Input:
            dof: DOF handle for joint
            force: Force applied
            environment: Environment index
        """

        boole = self.sim.gym.apply_dof_effort(self.envs[environment], dof, force)