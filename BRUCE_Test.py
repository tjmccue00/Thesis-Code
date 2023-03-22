import Environment as en
import Asset as ast
import Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
import keyboard
import BRUCE

model_file = r"/home/tjmccue/Documents/Thesis/BRUCE URDF/urdf"
model_file_name = r"BRUCE URDF v2.urdf"

gym = gymapi.acquire_gym()

sim = sm.Simulation(gym, dt=1/180)
sim.initialize()
sim.create_ground()

asset = ast.Asset(model_file, model_file_name, sim, gym, "BRUCE", rotat_t=[0, 90, -90], fix_base=True, trans_t=[0,0.5,0])
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

counter = 0
move = False
counter = 0

while not gym.query_viewer_has_closed(sim.get_Camera()):

    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    hip = 0
    knee = 0
    time_take = 0.05

    current_knee = brucie.normalize_joints(env.asset.joint_pos[1])
    current_hip = brucie.normalize_joints(env.asset.joint_pos[0])

    if keyboard.is_pressed('w'):
        move = True
        knee, hip = brucie.inverse_kin(0.1, -0.05)
        
        #kneeDis = float(input("Enter knee distance (degrees): "))*math.pi/180
        #hipDis = float(input("Enter hip distance (degrees): "))*math.pi/180

        kneeDis = knee - current_knee
        hipDis = hip - current_hip

        kneeDis = kneeDis/(time_take/sim.dt)
        hipDis = hipDis/(time_take/sim.dt)
        
    
    elif keyboard.is_pressed('s'):
        move = True
        knee, hip = brucie.inverse_kin(0.28, -0.05)
        kneeDis = knee - current_knee
        hipDis = hip - current_hip

        kneeDis = kneeDis/(time_take/sim.dt)
        hipDis = hipDis/(time_take/sim.dt)

    if move:
        
        env.asset.joint_pos[0] += hipDis
        env.asset.joint_pos[2] += hipDis
        env.asset.joint_pos[4] += -hipDis
        env.asset.joint_pos[6] += -hipDis

        env.asset.joint_pos[1] += kneeDis
        env.asset.joint_pos[3] += kneeDis
        env.asset.joint_pos[5] += kneeDis
        env.asset.joint_pos[7] += -kneeDis

        if counter-1 > time_take/sim.dt:
            move = False
            counter = 0

        if counter % 60:
            print("Knee Dis: ", kneeDis*180/math.pi*(time_take/sim.dt))
            print("Hip Dis: ", hipDis*180/math.pi*(time_take/sim.dt))
            print("Current Knee Pos: ", brucie.normalize_joints(env.asset.joint_pos[1])*180/math.pi)
            print("Current Hip Pos: ", brucie.normalize_joints(env.asset.joint_pos[0])*180/math.pi)
            print("")
        counter += 1


    gym.set_actor_dof_states(env.envs[0], env.actors[0], asset.joint_state, gymapi.STATE_ALL)

    env.render()

sim.end_sim()