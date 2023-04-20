import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from Networks import Actor, Critic
from ReplayBuffer import ReplayBuffer
from OUNoise import OrnsteinUhlenbeckNoise

class Agent(object):
    
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = rb.ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        
        self.actor = ac.Actor(alpha, n_actions, 'Actor', input_dims, self.sess, layer1_size, layer2_size, env.action_space)

        self.critic = cr.Critic(beta, n_actions, 'Critic', input_dims, self.sess, layer1_size, layer2_size)

        self.target_actor = ac.Actor(alpha, n_actions, 'TargetActor', input_dims, self.sess, layer1_size, layer2_size, env.action_space)

        self.target_critic = cr.Critic(beta, n_actions, 'TargetCritic', input_dims, self.sess, layer1_size, layer2_size)

        self.noise = oun.OrnsteinUhlenbeckNoise(mu=np.zeros(n_actions))
        
        self.update_critic = [self.target_critic.params[i].assign(tf.multiply(self.critic_params[i], self.tau) + tf.multiply(self.target_critic_params[i], 1 - self.tau)) for i in range(len(self.target_critic.params))]

        self.update_actor = [self.target_actor.params[i].assign(tf.multiply(self.actor_params[i], self.tau) + tf.multiply(self.target_actor_params[i], 1 - self.tau)) for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau

        else:
            self.target_critic.ses.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)


        critic_value = self.target_critic.predict(new_state, self.target_actor.predict(new_state))

        target = []

        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value[j]*done[j])
        
        target = np.reshape(target, (self.batch_size, 1))
        _ = self.critic.train(state, action, target)

        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
