import numpy as np

class qlearning():

    def __init__(self, actions, states, action_bounds, state_bounds, learn_rate, gamma, bin_size):
        self.actions = actions
        self.action_bounds = action_bounds
        self.states = states
        self.state_bounds = state_bounds
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.bin_size = bin_size
        
        self.bins = []
        self.qtable = None
        

    def initialize_table(self):
        for i in range(len(self.states)):
                self.bins.append(np.linspace(self.state_bounds[i][0], self.state_bounds[i][1], self.bin_size))

        self.qtable = np.random.uniform(low=-1, high=1, size=[self.bin_size]* self.states + [self.actions])

    def discretize(self, state):
        disc_states = []
        for i in range(len(state)):
            disc_states.append(np.digitize(state[i], self.bins[i]) - 1)

    def get_action(self):
        pass