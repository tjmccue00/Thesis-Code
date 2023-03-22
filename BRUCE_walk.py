import Environment as en
import Asset as ast
import Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import keyboard
import BRUCE
import gait

model_file = r"/home/tjmccue/Documents/Thesis/BRUCE URDF/urdf"
model_file_name = r"BRUCE URDF v2.urdf"

gym = gymapi.acquire_gym()

sim = sm.Simulation(gym, dt=1/180)
sim.initialize()
sim.create_ground()

asset = ast.Asset(model_file, model_file_name, sim, gym, "BRUCE", rotat_t=[0, 90, -90], fix_base=False, trans_t=[0,0.5,0])
asset.initialize()
asset.configure_joint_lims()

env = en.Environment(asset, sim, gym, 1, 1, 5)
env.initialize()

stiffness = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
damping = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
friction = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
max_velo = (100, 100, 100, 100, 100, 100, 100, 100)
max_effort = (800, 800, 800, 800, 800, 800, 800, 800)
env.set_actor_dof_props(stiffness, damping, friction, max_effort, max_velo)

env.set_Render(True)
env.set_Sync(True)

brucie = BRUCE.Bruce(0.14, 0.16, 0.2, [1,1,1,1])

cam_pos = gymapi.Vec3(0, 0.65, 1.5)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(sim.get_Camera(), None, cam_pos, cam_target)

xy = brucie.kin(env.asset.joint_pos[1], env.asset.joint_pos[0])
init_feet_pos = [[xy[1], xy[0]], [xy[1], xy[0]], [xy[1], xy[0]], [xy[1], xy[0]]]
phase = [(0.28, 0.0), (0.28, 0.075), (0.25, -0.0), (0.28, -0.075)] #3-0.25
walk = gait.Gait(sim.dt, 1, phase, [0, 1, 2, 3], init_feet_pos)

walk_now = False
while not gym.query_viewer_has_closed(sim.get_Camera()):

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    if walk_now:
        feet_pos = walk.run()
        knee1, hip1 = brucie.inverse_kin(feet_pos[1][0], feet_pos[1][1])
        knee2, hip2 = brucie.inverse_kin(feet_pos[0][0], feet_pos[0][1])
        knee3, hip3 = brucie.inverse_kin(feet_pos[2][0], feet_pos[2][1])
        knee4, hip4 = brucie.inverse_kin(feet_pos[3][0], feet_pos[3][1])
        env.asset.joint_pos[0] = -hip1
        env.asset.joint_pos[1] = -knee1
        env.asset.joint_pos[2] = hip2
        env.asset.joint_pos[3] = knee2
        env.asset.joint_pos[4] = -hip3
        env.asset.joint_pos[5] = knee3
        env.asset.joint_pos[6] = hip4
        env.asset.joint_pos[7] = knee4


    if keyboard.is_pressed('s'):
        walk_now = True
     

    gym.set_actor_dof_states(env.envs[0], env.actors[0], asset.joint_state, gymapi.STATE_ALL)

    env.render()

sim.end_sim()