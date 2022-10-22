


class RL_Frame:


    def __init__(self, model, environment, name, agent, algorithms):
        self.model = model
        self.environment = environment
        self.model_name = name
        self.reward_data = []
        self.agent = agent
        self.algorithms = algorithms
        self.curr_algorithm = algorithms[0]
        self.curr_runData = []

    def run(self):
        states = self.environment.getStates();
        
