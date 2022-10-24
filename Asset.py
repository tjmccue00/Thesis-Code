import math
import numpy as np
from isaacgym import gymapi, gymutil


class Asset():

    def __init__ (self, root_file, asset_name, sim, gym,\
                fix_base = True,\
                mesh_materials = True,\
                collapse_fixed = True,\
                thickness = 0.001,\
                drive_mode = gymapi.DOF_MODE_POS):

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

    def instantiate(self):

        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = self.fix_base
        asset_opts.use_mesh_materials = self.mesh_materials
        asset_opts.collapse_fixed_joints = self.collapse_fixed
        asset_opts.default_dof_drive_mode = self.drive_mode
        asset_opts.thickness = self.th

        self.asset = (self.gym).load_asset(self.sim, self.root_file, self.asset_name, asset_opts)
