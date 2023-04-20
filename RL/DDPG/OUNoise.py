# target q network class
# replay buffer class
# use batch norm
# stochastic policy used to learn greedy policy
# two actor and critic networks a target for each
# updates are soft, according to theta prime = tau*theta + (1-tau)*theta_prime tau << 1
# target actor is just the evaluation actor plus some noise process
# used ornstein uhlenbeck process -> need class for noise
import os
import numpy as np
import tensorflow as tf


class OrnsteinUhlenbeckNoise(object):

    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.x0 = x0
        self.reset()


    def __call__(self):
        x = self.x_prev + self.theta*(self.mu-self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

