import numpy as np

def setup(self):
    np.random.seed()

def act(self, game_state: dict):
    self.logger.info('From My_agent_1: act')
    self.logger.info(game_state)
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    # return game_state['user_input']
