import numpy as np

class qlearning():

    def __init__(self):
        self.actions = 2
        self.states = 4
        self.learn_rate = 0.15
        self.gamma = 0.9
        
        self.qtable = None
        

    def discretize(self, state):
        pass

    def get_action(self):
        pass