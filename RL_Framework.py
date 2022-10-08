


class RL_Frame:


    def __init__(self, model, environment, name, agent):
        self.model = model
        self.environment = environment
        self.model_name = name
        self.reward_data = []
        self.agent = agent
