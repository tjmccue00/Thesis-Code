from RL.DDPG.DDPG import Agent
from Environments.BruceEnv import BruceEnv
import numpy as np

if __name__ == "__main__":
    be = BruceEnv()

    multiplier = 10*be.sim.dt
    print(multiplier)
    agent = Agent(input_dims=(15,), n_actions=8, act_mult=multiplier, min_action=-multiplier, max_action=multiplier)
    n_games = 5000

    best_score = 0
    score_history = []
    load_checkpoint = False

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
        avg_score = np.mean(score_history[-100:])
        print('episode ', i, 'score %.1f' % avg_score)
   