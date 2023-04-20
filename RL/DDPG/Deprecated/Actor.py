import os
import numpy as np
import tensorflow as tf
from tensorflow.initializers import random_uniform

class Actor(object):

    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir=''):

        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.chkpt_dir = chkpt_dir
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient())

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape = [None, *self.input_dims], name='inputs')

            self.action_gradient = tf.placeholder(tf.float32, shape=[None, self.n_actions])

            f1 = 1 / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims, kernel_initializer=random_uniform(-f1, f1), bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activiation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(self.input, units=self.fc2_dims, kernel_initializer=random_uniform(-f2, f2), bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activiation = tf.nn.relu(batch2)

            f3 = 0.003
            mu = tf.layers.dense(layer2_activation, units=self.n_actions, activation='tanh', kernel_initializer=random_uniform, (-f3, f3), bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):

        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action_gradient: gradients})

    def save_checkpoint(self):
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        self.saver.restore(self.sess, self.checkpoint_file)