import math
import numpy as np
from isaacgym import gymapi, gymutil

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

model_file = r"/home/tjmccue/Documents/Thesis/Cart and Pendulum Model/URDF Model v2/Cart_and_Pendelum_Assembly/urdf/"
model_file_name = r"Cart and Pendelum Assembly.urdf"

gym = gymapi.acquire_gym()

sim_Params = gymapi.SimParams()
dt = 1.0/60
use_GPU = True
GPU_pipeline = False
sim_Params.physx.solver_type = 1
sim_Params.physx.num_position_iterations = 6
sim_Params.physx.num_position_iterations = 1
sim_Params.physx.use_gpu = use_GPU
sim_Params.dt = dt

sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_Params)

if sim is None:
    print("Couldn't create simulation")
    quit()

plane_parameters = gymapi.PlaneParams()
gym.add_ground(sim, plane_parameters)

camera = gym.create_viewer(sim, gymapi.CameraProperties())
if camera is None:
    print("Couldn't create camera")
    quit()

asset_opts = gymapi.AssetOptions()
asset_opts.fix_base_link = True
asset_opts.use_mesh_materials = True
asset_opts.collapse_fixed_joints = True
asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_opts.thickness = 0.001


asset = gym.load_asset(sim, model_file, model_file_name, asset_opts)

joint_names = gym.get_asset_dof_names(asset)
joint_props = gym.get_asset_dof_properties(asset)

num_joints = gym.get_asset_dof_count(asset)
joint_state = np.zeros(num_joints, dtype=gymapi.DofState.dtype)

joint_types = [gym.get_asset_dof_type(asset, i) for i in range(num_joints)]
joint_pos = joint_state['pos']

has_lim = joint_props['hasLimits']
low_lim = joint_props['lower']
up_lim = joint_props['upper']

defaults = np.zeros(num_joints)
speeds = np.zeros(num_joints)
for i in range(num_joints):
    if has_lim[i]:
        if joint_types[i] == gymapi.DOF_ROTATION:
            low_lim[i] = clamp(low_lim[i], -math.pi, math.pi)
            up_lim[i] = clamp(up_lim[i], -math.pi, math.pi)
        if low_lim[i] > 0:
            defaults[i] = low_lim[i]
        elif up_lim[i] < 0:
            defaults[i] = up_lim[i]
    else:
        if joint_types[i] == gymapi.DOF_ROTATION:
            low_lim[i] = -math.pi/2
            up_lim[i] = math.pi/2
        elif joint_types[i] == gymapi.DOF_TRANSLATION:
            low_lim[i] = -0.5
            up_lim[i] = 0.5
    joint_pos[i] = defaults[i]
    if joint_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = 0.5 * clamp(2 * (up_lim[i] - low_lim[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = 0.5 * clamp(2 * (up_lim[i] - low_lim[i]), 0.1, 5.0)

env_lower = gymapi.Vec3(-1, 0.0, -1)
env_upper = gymapi.Vec3(1, 1, 1)
environ = gym.create_env(sim, env_lower, env_upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0, 0.0)
pose.r = gymapi.Quat(0.7071, 0, 0, -0.7071)
actor_handle = gym.create_actor(environ, asset, pose)

gym.set_actor_dof_states(environ, actor_handle, joint_state, gymapi.STATE_ALL)

ANIM_LOWER = 1
ANIM_UPPER = 2
ANIM_DEFAULT = 3
ANIM_FINISHED = 4

animation = ANIM_LOWER
current_joint = 0

while not gym.query_viewer_has_closed(camera):

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    speed = speeds[current_joint]

    if animation == ANIM_LOWER:
        joint_pos[current_joint] -= speed * dt
        if joint_pos[current_joint] <= low_lim[current_joint]:
            joint_pos[current_joint] = low_lim[current_joint]
            animation = ANIM_UPPER
    elif animation == ANIM_UPPER:
        joint_pos[current_joint] += speed * dt
        if joint_pos[current_joint] >= up_lim[current_joint]:
            joint_pos[current_joint] = up_lim[current_joint]
            animation = ANIM_DEFAULT
    elif animation == ANIM_DEFAULT:
        joint_pos[current_joint] -= speed * dt
        if joint_pos[current_joint] <= defaults[current_joint]:
            joint_pos[current_joint] = defaults[current_joint]
            animation = ANIM_FINISHED
    elif animation == ANIM_FINISHED:
        joint_pos[current_joint] = defaults[current_joint]
        current_joint = (current_joint + 1) % num_joints
        animation = ANIM_LOWER

    gym.set_actor_dof_states(environ, actor_handle, joint_state, gymapi.STATE_POS)

    gym.step_graphics(sim)
    gym.draw_viewer(camera, sim, True)

    gym.sync_frame_time(sim)

gym.destroy_viewer(camera)
gym.destroy_sim(sim)
