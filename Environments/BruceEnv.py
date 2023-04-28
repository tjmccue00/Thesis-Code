import Environments.SimModules.Environment as en
import Environments.SimModules.Asset as ast
import Environments.SimModules.Simulation as sm
from Environments.AssetClass.BRUCE import Bruce
import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
import random as rng
from scipy.spatial.transform import Rotation
import warnings


class BruceEnv(object):

    def __init__(self):
        self.model_file = r"/home/tjmccue/Documents/Thesis/BRUCE URDF/urdf"
        self.model_file_name = r"BRUCE URDF v2.urdf"

        self.gym = gymapi.acquire_gym()

        self.sim = sm.Simulation(self.gym, dt=1/180)
        self.sim.initialize()
        self.sim.create_ground()

        self.asset = ast.Asset(self.model_file, self.model_file_name, self.sim, self.gym, "BRUCE", rotat_t=[0, 90, -90], fix_base=False, trans_t=[0,0.3,0])
        self.asset.initialize()
        self.asset.configure_joint_lims()

        self.env = en.Environment(self.asset, self.sim, self.gym, 1, 1, 5)
        self.env.initialize()

        stiffness = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        damping = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        friction = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        max_velo = (100, 100, 100, 100, 100, 100, 100, 100)
        max_effort = (800, 800, 800, 800, 800, 800, 800, 800)
        self.env.set_actor_dof_props(stiffness, damping, friction, max_effort, max_velo)

        self.env.set_Render(True)
        self.env.set_Sync(True)

        self.brucie = Bruce(0.14, 0.16, 0.2, [1,1,1,1])

        cam_pos = gymapi.Vec3(0, 0.65, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.sim.get_Camera(), None, cam_pos, cam_target)

        self.hip1_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Hip_One")
        self.knee1_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Knee_One")
        self.hip2_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Hip_Two")
        self.knee2_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Knee_Two")
        self.hip3_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Hip_Three")
        self.knee3_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Knee_Three")
        self.hip4_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Hip_Four")
        self.knee4_handle = self.gym.find_actor_dof_handle(self.env.envs[0], self.env.actors[0], "Knee_Four")

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim.sim)
        self.actor_root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor)

        self.prev_xy = (0,0)

        self.first_reset = True
        self.first_step = True
        self.output_clock = 0
        self.stuck_ctr = 0

        warnings.simplefilter("ignore", UserWarning)
        print()



    def reset(self):

        if self.first_reset:
            self.first_reset = False
        else:
            self.env.asset.joint_pos[0] = 0
            self.env.asset.joint_pos[1] = 0
            self.env.asset.joint_pos[2] = 0
            self.env.asset.joint_pos[3] = 0
            self.env.asset.joint_pos[4] = 0
            self.env.asset.joint_pos[5] = 0
            self.env.asset.joint_pos[6] = 0
            self.env.asset.joint_pos[7] = 0

            self.gym.set_actor_dof_states(self.env.envs[0], self.env.actors[0], self.asset.joint_state, gymapi.STATE_ALL)
            bool = self.gym.set_actor_root_state_tensor(self.sim.sim, gymtorch.unwrap_tensor(self.saved_root_tensor))

        rigid_body_pos = self.gym.get_actor_rigid_body_states(self.env.envs[0], self.env.actors[0], gymapi.STATE_ALL)

        xyz = rigid_body_pos[0][0][0]
        rpy = rigid_body_pos[0][0][1]
        xyz_dot = rigid_body_pos[0][1][0]
        rpy_dot = rigid_body_pos[0][1][1]
        

        observations = (0, 0, 0, 0, 0, 0, 0, 0,  xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], rpy[3])
        self.output_clock = 0
        self.stuck_ctr = 0

        return observations


    def step(self, actions, verbose=False):


        hip1_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip1_handle)))
        knee1_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee1_handle)))
        hip2_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip2_handle)))
        knee2_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee2_handle)))
        hip3_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip3_handle)))
        knee3_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee3_handle)))
        hip4_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip4_handle)))
        knee4_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee4_handle)))

        
        self.env.asset.joint_pos[0] = -actions[6] + hip4_pos
        self.env.asset.joint_pos[1] = -actions[7] - knee4_pos
        self.env.asset.joint_pos[2] = actions[0] - hip1_pos
        self.env.asset.joint_pos[3] = actions[1] - knee1_pos
        self.env.asset.joint_pos[4] = -actions[4] - hip3_pos
        self.env.asset.joint_pos[5] = actions[5] + knee3_pos
        self.env.asset.joint_pos[6] = actions[2] + hip2_pos
        self.env.asset.joint_pos[7] = actions[3] + knee2_pos

        self.gym.set_actor_dof_states(self.env.envs[0], self.env.actors[0], self.asset.joint_state, gymapi.STATE_ALL)
        self.gym.simulate(self.sim.sim)
        self.gym.fetch_results(self.sim.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim.sim)
        if self.first_step:
            self.saved_root_tensor = self.actor_root_state_tensor.clone()
            self.first_step = False


        hip1_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip1_handle)))
        knee1_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee1_handle)))
        hip2_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip2_handle)))
        knee2_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee2_handle)))
        hip3_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip3_handle)))
        knee3_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee3_handle)))
        hip4_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.hip4_handle)))
        knee4_pos = float(self.brucie.normalize_joints(self.gym.get_joint_position(self.env.envs[0], self.knee4_handle)))

        rigid_body_pos = self.actor_root_state_tensor.clone()

        xyz = rigid_body_pos[:,0:3]
        rpy = rigid_body_pos[:,3:7]
        xyz_dot = rigid_body_pos[:,7:10]
        rpy_dot = rigid_body_pos[:,10:13]

        if verbose:
            roll, pitch, yaw = self.outputData((xyz[0], rpy[0]), (xyz_dot[0], rpy_dot[0]))

        x = xyz[0][0]
        y = xyz[0][2]
        z = xyz[0][1]
        x_dot = xyz_dot[0][0]
        y_dot = xyz_dot[0][1]

        base_lin_vel = quat_rotate_inverse(rpy, xyz_dot)
        base_ang_vel = quat_rotate_inverse(rpy, rpy_dot)

        reward = 2*(x - self.prev_xy[0]) + -0.5*(y - self.prev_xy[1]) + base_lin_vel[0][0] - 0.5*abs(base_lin_vel[0][1]) - 0.5*abs(base_lin_vel[0][2]) + \
                  -0.1*abs(base_ang_vel[0][0]) -0.1*abs(base_ang_vel[0][1]) -0.1*abs(base_ang_vel[0][2]) + 25*self.sim.dt
        
        reward = np.clip(reward, 0, None)
        done = False
        stuck = False
        if abs(x - self.prev_xy[0]) < 0.01:
            self.stuck_ctr += 1
            if self.stuck_ctr > 10/self.sim.dt:
                stuck = True
        if abs(hip1_pos*180/math.pi) > 180 or abs(hip2_pos*180/math.pi) > 180 or abs(hip3_pos*180/math.pi) > 180 or abs(hip4_pos*180/math.pi) > 180 or \
           abs(knee1_pos*180/math.pi) > 120 or abs(knee2_pos*180/math.pi) > 120 or abs(knee3_pos*180/math.pi) > 120 or abs(knee4_pos*180/math.pi) > 120 or \
            z < 0.1 or stuck or z > 0.55:
            done = True

        

        observations = (-hip1_pos, -knee1_pos, hip2_pos, knee2_pos, hip3_pos, knee3_pos, -hip4_pos, knee4_pos, \
                        xyz[0][0].item(), xyz[0][1].item(), xyz[0][2].item(), rpy[0][0].item(), rpy[0][1].item(), rpy[0][2].item(), rpy[0][3].item())

        return observations, reward, done
        
        

    def render(self):
        self.env.render()

    def close(self):
        self.sim.end_sim()

    def outputData(self, pos, vel):

        w = round(pos[1][3].item(), 2)
        x = round(pos[1][0].item(), 2)
        y = round(pos[1][1].item(), 2)
        z = round(pos[1][2].item(), 2)

        quat = Rotation.from_quat((x, y, z, w))
        rpy = quat.as_euler("xyz", degrees=True)
        roll = rpy[0] + 90
        pitch = rpy[1] + 90
        yaw = rpy[2]

        output = "X Pos: {:.2f} |Y Pos: {:.2f} |Z Pos: {:.2f}".format(pos[0][0], pos[0][2], pos[0][1]) + "     " \
                #+ "|Roll X: {:.2f} |Pitch Y: {:.2f} |Yaw Z : {:.2f}          ".format(pitch, roll, yaw)
        
        if self.output_clock % int(1/(10*self.sim.dt)) == 0:
            print(output, end="\r")
            if self.output_clock > 1*10**5:
                self.output_clock = 0
        self.output_clock += 1
        return pitch, roll, yaw