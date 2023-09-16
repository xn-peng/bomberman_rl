import numpy as np
import os
import pickle
import torch
from agent_code.my_agent.lstm import LSTMModel
from agent_code.tpl_agent.callbacks import state_to_features



def __init__(self):
    self.learning_rate = 0.01
    self.setup()

def setup(self):
    self.logger.info("Setting up model from scratch.")
    self.model = LSTMModel(17*17, 32, 1, 5)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.model.train()

def act(self, game_state: dict):
    self.logger.info('Picking action based on model predictions.')
    actions = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT']
    map = state_to_features(game_state).reshape(1, 1, -1)
    map = torch.tensor(map)
    probability = self.model(map)
    action = actions[torch.argmax(probability)]
    return action

def state_to_features(self, game_state: dict) -> np.array:
    if game_state is None:
        return None

    channels = game_state["field"]
    for x, y in game_state["coins"]:
        channels[x][y] = 3
    for other in game_state["others"]:
        if other[2]:
            x, y = other[3]
            channels[x][y] = 2
    x, y = game_state["self"][-1]
    channels[x][y] = 1

    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)

def save_model(self, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(self.model, file)

def load_model(self, file_path):
    with open(file_path, "rb") as file:
        self.model = pickle.load(file)

# Example usage:
# agent = myAgent()
# agent.save_model("my-saved-model.pt")
# agent.load_model("my-saved-model.pt")
