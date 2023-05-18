import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class Critic(keras.Model):
    
    def __init__(self, fc1_dims=512, fc2_dims=512, name='Critic', chkpt_dir =r'Learning Models/DDPG/'):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims=fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class Actor(keras.Model):
    
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512, name='Actor', chkpt_dir ='Learning Models/DDPG/'):
        super(Actor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims=fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        action_value = self.fc1(state)
        action_value = self.fc2(action_value)

        actions = self.mu(action_value)

        return actions

