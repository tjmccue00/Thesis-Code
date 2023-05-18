import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from RL.DDPG.Networks import Actor, Critic
from RL.DDPG.ReplayBuffer import ReplayBuffer
from RL.DDPG.OUNoise import OrnsteinUhlenbeckNoise

class Agent(object):
    
    def __init__(self, input_dims, tau=0.005, alpha=0.001, beta=0.002, gamma=0.99, 
                 n_actions=8, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64, min_action=-1, max_action=1, act_mult=1):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.min_action = min_action
        self.max_action = max_action
        self.act_mult = act_mult
        self.n_actions = n_actions
        
        self.actor = Actor(n_actions=n_actions, name='Actor', fc1_dims=layer1_size, fc2_dims=layer2_size)
        self.critic = Critic(name='Critic', fc1_dims=layer1_size, fc2_dims=layer2_size)

        self.target_actor = Actor(n_actions=n_actions, name='Target_Actor', fc1_dims=layer1_size, fc2_dims=layer2_size)
        self.target_critic = Critic(name='Target_Critic', fc1_dims=layer1_size, fc2_dims=layer2_size)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau == None:
            tau= self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action/self.act_mult, reward, new_state, done)

    def choose_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            #noise = self.noise()
            #mu_prime = actions + noise
            #print("Raw Actions")
            #print(actions)
            noise = tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)
            actions += noise
            actions = tf.clip_by_value(actions*self.act_mult, self.min_action, self.max_action)


        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        state = tf.convert_to_tensor(states, dtype=tf.float32)
        new_state = tf.convert_to_tensor(new_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            critic_value_ = tf.squeeze(self.target_critic(new_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
