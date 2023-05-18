from Environments.CartPoleEnv import CartPole
from RL.PPO.PPO import Agent
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


cp = CartPole()
N = 1000
batch_size = 750
n_epochs = 6
alpha = 3e-5

agent = Agent(n_actions=3, action_bounds=[-200,200], batch_size=batch_size, 
              alpha=alpha, n_epochs=n_epochs)

timestep=10
epochs =  75000
rewards = 0
solved = False 
steps = 0 
runs = [0]
data = {'score' : [0], 'avg' : [0]}
start = time.time()
ep = [i for i in range(0,epochs + 1,timestep)] 
learn_iters = 0
date = str(datetime.now())
bestScore = 0


for episode in range(1,epochs+1):
    observation = cp.reset()  # initial observation
    score = 0
    done = False
    temp_start = time.time()
    while not done and score <= 3600:
        #cp.render()
        ep_start = time.time()

        action, prob, val, action_idx = agent.choose_action(observation)
        observation_, reward, done = cp.step(action)
        steps += 1
        score += reward

        agent.store_memory(observation, action_idx, prob, val, reward, done)

        if steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    else:
        rewards += score
        runs.append(score)
    
    # Timestep value update
    if episode%timestep == 0:
        if round(np.mean(data['score'][-timestep:]),2) > bestScore:
            bestScore = round(np.mean(data['score'][-timestep:]),2)
            agent.save_models()
        print('Episode : {} | Reward -> {} | Max reward : {} |'.format(episode,round(np.mean(data['score'][-100:]),2), round(max(runs), 2)))
        data['score'].append(score)
        data['avg'].append(np.mean(data['score'][-100:]))
        rewards, runs= 0, []
        with open(agent.chkpt_dir+"PPO_data_" + date + ".csv", "a") as f:
            f.write(str(episode) + "," + str(round(np.mean(data['score'][-timestep:]),2)) + "\n")


        
plt.plot(ep, data['avg'], label = 'Avg')
plt.title('Average Reward v Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
        
cp.close()