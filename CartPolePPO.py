import gym
import numpy as np
from RL.PPO.PPO import Agent


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(n_actions=env.action_space.n, action_bounds=[0,1], batch_size=batch_size, 
                  alpha=alpha, n_epochs=n_epochs)

    n_games = 300

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
  
    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action, prob, val, action_idx = agent.choose_action(observation)
<<<<<<< Updated upstream
            observation_, reward, done, truncated, info = env.step(int(action))
=======
            observation_, reward, done, truncated, info = env.step(action)
>>>>>>> Stashed changes

            n_steps += 1
            score += reward
            agent.store_memory(observation, action_idx, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode: ', i, 'score %.1f' % score, 'avg. score %.1f' % avg_score, 
              'time steps', n_steps, 'learning steps', learn_iters)