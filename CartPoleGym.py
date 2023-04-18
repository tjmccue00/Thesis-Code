import gym
import QLearning as ql
import time
import numpy as np
import matplotlib.pyplot as plt

def Qtable(state_space,action_space,bin_size = 30):
    bins = []
    state_bounds = [[-4.8, 4.8], [-4,4], [-0.418, 0.418], [-4,4]]
    for i in range(state_space):
                bins.append(np.linspace(state_bounds[i][0], state_bounds[i][1], bin_size))
    
    q_table = np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space]))
    return q_table, bins

env = gym.make('CartPole-v1')
qtable = ql.qlearning(2, 4, [1,0], [[-4.8, 4.8], [-4,4], [-0.418, 0.418], [-4,4]], 0.2, 0.95, 30)

timestep=1000
epochs = 20000
rewards = 0
solved = False 
steps = 0 
runs = [0]
data = {'max' : [0], 'avg' : [0]}
start = time.time()
ep = [i for i in range(0,epochs + 1,timestep)] 
epsilon = 0.2
qtable.initialize_table()
qtab, bins = Qtable(len(env.observation_space.low), env.action_space.n)

for episode in range(1,epochs+1):
    current_state = qtable.discretize(env.reset()[0])  # initial observation
    score = 0
    done = False
    temp_start = time.time()
    while not done:
        steps += 1
        ep_start = time.time()
        if episode%timestep == 0:
                env.render()

        if np.random.uniform(0,1) < epsilon:
            action, action_idx = qtable.get_sample_action()
            action = int(action)
        else:
            action = np.argmax(qtab[current_state])
            #action, action_idx = qtable.get_action(current_state)

        observation, reward, done, truncated, info = env.step(action)
        next_state = qtable.discretize(observation)

        score += reward

        if not done:
            max_future_q = np.max(qtab[next_state])
            current_q = qtab[current_state+(action,)]
            new_q = (1-qtable.learn_rate)*current_q + qtable.learn_rate*(reward + qtable.gamma*max_future_q)
            qtab[current_state+(action,)] = new_q
             #qtable.update_table(next_state, current_state, action_idx, reward)

        current_state = next_state
    else:
        rewards += score
        runs.append(score)
        if score > 195 and steps >= 100 and solved == False: # considered as a solved:
            solved = True
            #print('Solved in episode : {} in time {}'.format(episode, (time.time()-ep_start)))
    
    # Timestep value update
    if episode%timestep == 0:
        print('Episode : {} | Reward -> {} | Max reward : {} | Time : {}'.format(episode,rewards/timestep, max(runs), time.time() - ep_start))
        data['max'].append(max(runs))
        data['avg'].append(rewards/timestep)
        if rewards/timestep >= 195: 
             pass
            #print('Solved in episode : {}'.format(episode))
        rewards, runs= 0, [0]
        
#plt.plot(ep, data['max'], label = 'Max')
plt.plot(ep, data['avg'], label = 'Avg')
plt.title('Average Reward v Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
#plt.legend(loc = "upper left")
plt.show()
        
env.close()