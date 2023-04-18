import numpy as np

class qlearning():
    """
    Sets up stores qlearning tables for implementing reinforcement algorithm easily
    """

    def __init__(self, num_actions, states, action_bounds, state_bounds, learn_rate, gamma, bin_size):
        self.action_space = num_actions
        self.actions = []
        self.action_bounds = action_bounds
        self.states = states
        self.state_bounds = state_bounds
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.bin_size = bin_size
        
        self.bins = []
        self.qtable = None
        

    def initialize_table(self):
        """
        Initializes QTable based on needed size of action space and state space
        """
        for i in range(self.states):
                self.bins.append(np.linspace(self.state_bounds[i][0], self.state_bounds[i][1], self.bin_size))

        self.qtable = np.random.uniform(low=-1, high=1, size=[self.bin_size]* self.states + [self.action_space]) 
        self.actions = np.linspace(self.action_bounds[0], self.action_bounds[1], self.action_space)

    def discretize(self, state):
        """
        Gets discrete state based upon given continuous states

        Input: List of states that is number of states in length
        Return: tuple of the discretized states
        """

        disc_states = []
        for i in range(len(state)):
            disc_states.append(np.digitize(state[i], self.bins[i]) - 1)
        return tuple(disc_states)

    def get_disc_act(self, action):
        disc_act = np.digitize(action, self.actions) - 1
        return disc_act

    def get_action(self, state):
        """
        Gets best action based on given states

        Input: List of continuous states
        Return: Tuple of action and discretized action
        """
        dig_stat = self.discretize(state)

        action_idx = np.argmax(self.qtable[dig_stat])
        action = self.actions[action_idx]

        return (action, action_idx)

    def get_sample_action(self):
        """
        Gets random action

        Input: List of continuous states
        Return: Tuple of action and discretized action
        """
        numb = np.random.randint(low=0, high=self.action_space, size=1)
        numb = numb[0]

        return (self.actions[numb], numb)

    def update_table(self, next_state, current_state, action_idx, reward):
        """
        Updates QTable based on the reward given from current action

        Input:
            next_state: state that arises from last action
            current_state: state that came before last action
            action_idx: discreteized action
            reward: Reward based on environment
        """

        #next_dig_stat = self.discretize(next_state)
        #current_dig_stat = self.discretize(current_state)
        max_future_q = np.max(self.qtable[next_state])
        current_q = self.qtable[current_state+(action_idx, )]
        new_q = (1 - self.learn_rate)*current_q + self.learn_rate*(reward + self.gamma*max_future_q)
        self.qtable[current_state+(action_idx,)] = new_q

if __name__ == "__main__":

    qtable = qlearning(3,2,[-2,2], [[-5, 5], [-2,2]], 0.1, 0.95, 10)

    qtable.initialize_table()
    state = [-3, 1.45]
    action, action_idx = qtable.get_action(state)

    qtable.update_table(state, action_idx, 100)
 