import Environments.SimModules.Environment as en
import Environments.SimModules.Asset as ast
import Environments.SimModules.Simulation as sm
import math
import numpy as np
from isaacgym import gymapi, gymutil
from Environments.BruceEnv import BruceEnv
import Environments.AssetClass.BRUCE as BRUCE
import Environments.AssetClass.gait as gait


be = BruceEnv()
done = False

observations = be.reset()

brucie = BRUCE.Bruce(0.14, 0.16, 0.2, [1,1,1,1])
xy = brucie.kin(observations[1], observations[0])
init_feet_pos = [[xy[1], xy[0]], [xy[1], xy[0]], [xy[1], xy[0]], [xy[1], xy[0]]]
phase = [(0.28, 0.0), (0.28, 0.075), (0.25, -0.0), (0.28, -0.075)]
walk = gait.Gait(be.sim.dt, 0.75, phase, [0, 1, 2, 3], init_feet_pos)
t = 0
already_reset = False

while not be.gym.query_viewer_has_closed(be.sim.get_Camera()):

    be.render()
    if not done:
        
        feet_pos = walk.run()
        
        knee1, hip1 = brucie.inverse_kin(feet_pos[1][0], feet_pos[1][1])
        knee2, hip2 = brucie.inverse_kin(feet_pos[0][0], feet_pos[0][1])
        knee3, hip3 = brucie.inverse_kin(feet_pos[2][0], feet_pos[2][1])
        knee4, hip4 = brucie.inverse_kin(feet_pos[3][0], feet_pos[3][1])
        

        hip1_action = hip1 - observations[0]
        knee1_action = knee1 - observations[1]
        hip2_action = hip2 - observations[2]
        knee2_action = knee2 - observations[3]
        hip3_action = hip3 - observations[4]
        knee3_action = knee3 - observations[5]
        hip4_action = hip4 - observations[6]
        knee4_action = knee4 - observations[7]

        actions = (hip1_action, knee1_action, hip2_action, knee2_action, hip3_action, knee3_action, hip4_action, knee4_action)

        if already_reset:
            actions = (0, 0, 0, 0, 0, 0, 0, 0)

        observations, reward, done = be.step(actions)
        t += be.sim.dt

        if t > 10:
            done = True
            t = 0

    else:
        be.reset()
        done = False
        already_reset = True