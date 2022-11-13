import math
import numpy as np
from isaacgym import gymapi, gymutil

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

class Asset():

    def __init__ (self, root_file, asset_name, sim, gym, name,\
                fix_base = True,\
                mesh_materials = True,\
                collapse_fixed = True,\
                thickness = 0.001,\
                drive_mode = gymapi.DOF_MODE_POS,\
                rotat_t = [0,0,0],\
                trans_t = [0,0,0]):

            self.name = name
            self.root_file = root_file
            self.asset_name = asset_name
            self.sim = sim
            self.gym = gym
            self.fix_base = fix_base
            self.mesh_materials = mesh_materials
            self.collapse_fixed = collapse_fixed
            self.th = thickness
            self.drive_mode = drive_mode
            self.asset = None

            self.joint_names = []
            self.joint_props = []
            self.num_joints = 0
            self.joint_state = []
            self.has_lim = []
            self.low_lim = []
            self.up_lim = []
            self.defaults = []
            self.rotat_t = rotat_t
            self.trans_t = trans_t

    def initialize(self):

        asset_opts = gymapi.AssetOptions()
        vhacd_para = gymapi.VhacdParams()
        asset_opts.fix_base_link = self.fix_base
        asset_opts.use_mesh_materials = self.mesh_materials
        asset_opts.collapse_fixed_joints = self.collapse_fixed
        asset_opts.default_dof_drive_mode = self.drive_mode
        asset_opts.thickness = self.th
        asset_opts.vhacd_enabled = True
        #vhacd_para.mode = 1
        #vhacd_para.resolution = 1000000
        #asset_opts.vhacd_params = vhacd_para
        self.asset = self.gym.load_asset(self.sim.sim, self.root_file, self.asset_name, asset_opts)

        self.joint_names = self.gym.get_asset_dof_names(self.asset)
        self.joint_props = self.gym.get_asset_dof_properties(self.asset)
        self.num_joints = self.gym.get_asset_dof_count(self.asset)
        self.joint_state = np.zeros(self.num_joints, dtype=gymapi.DofState.dtype)
        self.joint_types = [self.gym.get_asset_dof_type(self.asset, i) for i in range(self.num_joints)]
        self.joint_pos = self.joint_state['pos']
        self.has_lim = self.joint_props['hasLimits']
        self.low_lim = self.joint_props['lower']
        self.up_lim = self.joint_props['upper']

        self.defaults = np.zeros(self.num_joints)

    def configure_joint_lims(self):

        for i in range(self.num_joints):
            if self.has_lim[i]:
                if self.joint_types[i] == gymapi.DOF_ROTATION:
                    self.low_lim[i] = clamp(self.low_lim[i], -math.pi, math.pi)
                    self.up_lim[i] = clamp(self.up_lim[i], -math.pi, math.pi)
                if self.low_lim[i] > 0:
                    self.defaults[i] = self.low_lim[i]
                elif self.up_lim[i] < 0:
                    self.defaults[i] = self.up_lim[i]
            else:
                if self.joint_types[i] == gymapi.DOF_ROTATION:
                    self.low_lim[i] = -math.pi/2
                    self.up_lim[i] = math.pi/2
                elif self.joint_types[i] == gymapi.DOF_TRANSLATION:
                   self.low_lim[i] = -0.5
                   self.up_lim[i] = 0.5
            self.joint_pos[i] = self.defaults[i]

    def print_joint_names(self):
        for i in range(len(self.joint_names)):
            print(self.joint_names[i])

    def get_joint_names(self):
        return self.joint_names


