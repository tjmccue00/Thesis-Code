import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import SGNN    

# pass in upper and lower bounds for each observation space

domain_ub = [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
domain_lb = [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
grid_size = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] # tunable/ number of neurons in each layer
n_actions = 6
dim = len(domain_ub)

layer_weight_initializers = { + 1: tf.initializers.ones()}

mu_grid, variance_arr = SGNN.cmptGaussCenterandWidth(domain_lb, domain_ub, grid_size)

actor = SGNN.GaussianNet(mu_grid, variance_arr, 12, 
                                weight_initializers=layer_weight_initializers)
# add dense layer with hyperbolic tangent
# pull parameters from network to look at convergence
critic = SGNN.GaussianNet(mu_grid, variance_arr, 1, 
                                weight_initializers=layer_weight_initializers)

target_actor = SGNN.GaussianNet(mu_grid, variance_arr, 12,
                                weight_initializers=layer_weight_initializers)

target_critic = SGNN.GaussianNet(mu_grid, variance_arr, 1, 
                                weight_initializers=layer_weight_initializers)

actor_output = keras.layers.Dense(n_actions, activation='tanh')

target_actor._output_layer=actor_output
actor._output_layer=actor_output

actor_inputs = tf.keras.layers.Input(shape=(dim,))
target_actor_inputs = tf.keras.layers.Input(shape=(dim,))
critic_inputs = tf.keras.layers.Input(shape=(dim,))
target_critic_inputs = tf.keras.layers.Input(shape=(dim,))

actor_outputs = actor(actor_inputs)
target_actor_outputs = target_actor(target_actor_inputs)
critic_outputs = actor(critic_inputs)
target_critic_outputs = target_actor(target_critic_inputs)

actor = tf.keras.models.Model(actor_inputs, actor_outputs)
target_actor = tf.keras.models.Model(target_actor_inputs, target_actor_outputs)
critic = tf.keras.models.Model(critic_inputs, critic_outputs)
target_actor = tf.keras.models.Model(target_critic_inputs, target_critic_outputs)

actor.summary()
critic.summary()

test = keras.models.Sequential()
test.add(keras.layers.Dense(300, input_shape=(11,), activation='relu'))
test.add(keras.layers.Dense(400, activation='relu'))
test.add(keras.layers.Dense(6, activation='tanh'))

test.summary()