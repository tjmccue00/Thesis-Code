import math
import numpy as np
from isaacgym import gymapi, gymutil

class Simulation():

    def __init__(self, gym,\
                dt = 1.0/60,\
                use_GPU = True,\
                GPU_pipeline = False,\
                solver_type = 1,\
                sim_type = gymapi.SimType.SIM_PHYSX,\
                num_position_iterations = 9,\
                num_velocity_iterations = 4):

        self.gym = gym
        self.dt = dt
        self.use_GPU = use_GPU
        self.GPU_pipe = GPU_pipeline
        self.solver_type = solver_type
        self.num_pos_iters = num_position_iterations
        self.num_vel_iters = num_velocity_iterations
        self.sim_type = sim_type
        self.sim = None
        self.camera = None

    def initialize(self):

            sim_params = gymapi.SimParams()
            sim_params.physx.solver_type = self.solver_type
            sim_params.physx.num_position_iterations = self.num_pos_iters
            sim_params.physx.num_velocity_iterations = self.num_vel_iters
            sim_params.physx.use_gpu = self.use_GPU
            sim_params.use_gpu_pipeline = self.GPU_pipe

            self.sim = self.gym.create_sim(0, 0, self.sim_type, sim_params)

            self.camera = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

    def create_ground(self):

        plane_parameters = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_parameters)

    def end_sim(self):

        self.gym.destroy_viewer(self.camera)
        self.gym.destroy_sim(self.sim)

    def get_Camera(self):
        return self.camera