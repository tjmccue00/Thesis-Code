from Environments.CartPoleEnv import CartPole
from RL.QLearning import Agent
import time
import numpy as np
import matplotlib.pyplot as plt


qtable = Agent(5, 4, [-200, 200], [[-.2095, .2095], [-10, 10], [-0.62, 0.62], [-10, 10]], 0.2, 0.995, 30)

timestep=500
epochs =  50000
rewards = 0
solved = False 
steps = 0 
runs = [0]
data = {'score' : [0], 'avg' : [0]}
start = time.time()
ep = [i for i in range(0,epochs + 1,timestep)] 
epsilon = 0.2
learn_iters = 0
qtable.initialize_table()
cp = CartPole()


for episode in range(1,epochs+1):
    current_state = qtable.discretize(cp.reset())  # initial observation
    score = 0
    done = False
    temp_start = time.time()
    while not done and score <= 3600:
        ep_start = time.time()

        if np.random.uniform(0,1) < epsilon:
            action, action_idx = qtable.get_sample_action()
            action = int(action)
        else:
            action = np.argmax(qtable.qtable[current_state])
            action, action_idx = qtable.get_action(current_state)
            action = int(action)

        next_state, reward, done = cp.step(action)
        steps += 1
  
        next_state = qtable.discretize(next_state)

        score += reward

        if not done:
            qtable.update_table(next_state, current_state, action_idx, reward)

        current_state = next_state
    else:
        rewards += score
        runs.append(score)
    
    # Timestep value update
    if episode%timestep == 0:
        print('Episode : {} | Reward -> {} | Max reward : {} | Time : {}'.format(episode,round(np.mean(data['score'][-100:]),2), max(runs), round(time.time() - ep_start, 3)))
        data['score'].append(score)
        data['avg'].append(np.mean(data['score'][-100:]))
        if rewards/timestep >= 195: 
             pass
            #print('Solved in episode : {}'.format(episode))
        rewards, runs= 0, [0]
qtable.save_qtable("Test3")
        
plt.plot(ep, data['avg'], label = 'Avg')
plt.title('Average Reward v Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
        
cp.close()