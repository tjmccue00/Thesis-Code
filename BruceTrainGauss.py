from RL.DDPG.DDPG import Agent
from Environments.BruceEnv import BruceEnv
import numpy as np
from Environments.AssetClass.BRUCE import Bruce
from Environments.AssetClass.gait import Gait
import tensorflow as tf
from datetime import datetime

if __name__ == "__main__":
    be = BruceEnv()

    multiplier = 20*be.sim.dt
    print(multiplier)
    agent = Agent(input_dims=(15,), n_actions=8, act_mult=multiplier, min_action=-multiplier, max_action=multiplier)
    n_games = 5000

    best_score = 0
    score_history = []
    load_checkpoint = False
    date = str(datetime.now())
    

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = be.reset()
            action = (0,0,0,0,0,0,0,0)
            observation_, reward, done = be.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    if False:
        observation = be.reset()
        brucie = Bruce(0.14, 0.16, 0.2, [1,1,1,1])
        xy = brucie.kin(observation[1], observation[0])
        init_feet_pos = [[xy[1], xy[0]], [xy[1], xy[0]], [xy[1], xy[0]], [xy[1], xy[0]]]
        phase = [(0.28, 0.0), (0.28, 0.075), (0.25, -0.0), (0.28, -0.075)]
        walk = Gait(be.sim.dt, 0.75, phase, [0, 1, 2, 3], init_feet_pos)

        for i in range(int(10/be.sim.dt)):
            

            feet_pos = walk.run()
            
            knee1, hip1 = brucie.inverse_kin(feet_pos[1][0], feet_pos[1][1])
            knee2, hip2 = brucie.inverse_kin(feet_pos[0][0], feet_pos[0][1])
            knee3, hip3 = brucie.inverse_kin(feet_pos[2][0], feet_pos[2][1])
            knee4, hip4 = brucie.inverse_kin(feet_pos[3][0], feet_pos[3][1])
            

            hip1_action = hip1 - observation[0]
            knee1_action = knee1 - observation[1]
            hip2_action = hip2 - observation[2]
            knee2_action = knee2 - observation[3]
            hip3_action = hip3 - observation[4]
            knee3_action = knee3 - observation[5]
            hip4_action = hip4 - observation[6]
            knee4_action = knee4 - observation[7]

            action = (hip1_action, knee1_action, hip2_action, knee2_action, hip3_action, knee3_action, hip4_action, knee4_action)
            observation_, reward, done = be.step(action)
            observation = observation_
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                    agent.learn()

    bestScore = float('-inf')
    for i in range(n_games):
        observation = be.reset()
        done = False
        score = 0
        while not done:
            if i > 4995:
                be.render()
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done = be.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])
        print('episode ', i, 'score %.1f' % avg_score)
        if i % 10:
            if avg_score > bestScore:
                agent.save_models()
                bestScore = avg_score
            with open(agent.actor.checkpoint_dir +"DDPG_data_" + date + ".csv", "a") as f:
                f.write(str(i) + "," + str(round(avg_score,2)) + "\n")