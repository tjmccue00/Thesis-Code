import gym
#from RL.QLearning import Agent
from RL.PPO.PPO import Agent
import time
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1')
#qtable = Agent(2, 4, [0,1], [[-4.8, 4.8], [-4,4], [-0.418, 0.418], [-4,4]], 0.2, 0.95, 30)

timestep=1
epochs = 300
rewards = 0
solved = False 
steps = 0 
runs = [0]
data = {'score' : [0], 'avg' : [0]}
start = time.time()
ep = [i for i in range(0,epochs + 1,timestep)] 
epsilon = 0.2
learn_iters = 0
#qtable.initialize_table()
ppo = Agent(n_actions=env.action_space.n, batch_size=5,
                  alpha=0.0003, n_epochs=4,
                  input_dims=env.observation_space.shape)

for episode in range(1,epochs+1):
    #current_state = qtable.discretize(env.reset()[0])  # initial observation
    current_state = env.reset()[0]
    score = 0
    done = False
    temp_start = time.time()
    while not done:
        ep_start = time.time()
        if episode%timestep == 0:
                env.render()

        #if np.random.uniform(0,1) < epsilon:
        #    action, action_idx = qtable.get_sample_action()
        #    action = int(action)
        #else:
        #    action = np.argmax(qtable.qtable[current_state])
        #    action, action_idx = qtable.get_action(current_state)
        #    action = int(action)

        action, prob, val = ppo.choose_action(current_state)
        next_state, reward, done, truncated, info = env.step(action)
        steps += 1
  
        ppo.store_memory(current_state, action, prob, val, reward, done)
        #next_state = qtable.discretize(observation)

        score += reward

        if steps % 20 == 0:
            ppo.learn()
            learn_iters += 1
            #qtable.update_table(next_state, current_state, action_idx, reward)

        current_state = next_state
    else:
        rewards += score
        runs.append(score)
        if score > 195 and steps >= 100 and solved == False: # considered as a solved:
            solved = True
    
    # Timestep value update
    if episode%timestep == 0:
        print('Episode : {} | Reward -> {} | Max reward : {} | Time : {}'.format(episode,np.mean(data['score'][-100:]), max(runs), time.time() - ep_start))
        data['score'].append(score)
        data['avg'].append(np.mean(data['score'][-100:]))
        if rewards/timestep >= 195: 
             pass
            #print('Solved in episode : {}'.format(episode))
        rewards, runs= 0, [0]
        
plt.plot(ep, data['avg'], label = 'Avg')
plt.title('Average Reward v Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
        
env.close()