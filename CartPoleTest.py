from Environments.CartPoleEnv import CartPole
from RL.QLearning import Agent
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from isaacgym import gymapi, gymutil
import random as rng

qtable = Agent(6, 4, [-200, 200], [[-.2095, .2095], [-10, 10], [-0.62, 0.62], [-10, 10]], 0.2, 0.95, 30)

qtable.load_qtable("Test1")

cp = CartPole()

start = False

done = False
current_state = cp.reset()
envum = 0

while not cp.gym.query_viewer_has_closed(cp.sim.get_Camera()):

    cp.render()
    if not done:
        current_state = qtable.discretize(current_state)
        action, action_idx = qtable.get_action(current_state)
        current_state, reward, done = cp.step(action)

    else:
        print('Reseting Environment ', envum)
        envum += 1
        current_state, reward, done = cp.step(0)
        cp.reset()



