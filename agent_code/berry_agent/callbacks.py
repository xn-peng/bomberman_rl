import numpy as np
import os, pickle
import torch
from agent_code.berry_agent.lstm import LSTMModel
learning_rate = 0.01

def setup(self):
    # print(self)
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")
    #     self.model = LSTMModel(17*17, 32, 1, 5)
    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)
    self.logger.info("Setting up model from scratch.")
    self.model = LSTMModel(17*17, 32, 1, 5)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    self.model.train()

def act(agent, game_state: dict):
    # print(agent)
    # raise "123"
    agent.logger.info('Pick action at random, but no bombs.')
    actions = ['RIGHT', 'LEFT', 'UP', 'DOWN', "WAIT"]
    map = state_to_features(game_state).reshape(1, 1, -1)
    map = torch.tensor(map)
    probability = agent.model(map)
    action = actions[torch.argmax(probability)]
    return action




def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = game_state["field"]
    for x, y in game_state["coins"]:
        channels[x][y] = 3
    for other in game_state["others"]:
        if other[2]:
            x, y = other[3]
            channels[x][y] = 2
    x, y = game_state["self"][-1]
    channels[x][y] = 1

    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
