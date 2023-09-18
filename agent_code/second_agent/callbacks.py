import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")
    #     weights = np.random.rand(len(ACTIONS))
    #     self.model = weights / weights.sum()
    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)
    #         self.logger.info(self.model)

    if os.path.isfile("agent_q_table.npy"):
        self.logger.info("Loading Q-table from saved file")
        self.q_table = np.load("agent_q_table.npy")
    else:
        self.logger.info("Q-table file not found, use random strategy")



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # self.logger.info(game_state)
    self.logger.info(self.q_table)

    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])


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

        # # For example, you could construct several channels of equal shape, ...
        # channels = []
        # channels.append(...)
        # # concatenate them as a feature tensor (they must have the same shape), ...
        # stacked_channels = np.stack(channels)
        # # and return them as a vector
        # return stacked_channels.reshape(-1)
        # Extract features from the game board (field)

    field = game_state['field']
    agent_position = game_state['self'][3]

    # Define a feature vector
    feature_vector = []

    # Extract features based on agent's position
    row, col = agent_position

    # Feature 1: Number of empty tiles around the agent (up, down, left, right)
    empty_tiles = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        r, c = row + dr, col + dc
        if 0 <= r < field.shape[0] and 0 <= c < field.shape[1] and field[r, c] == 0:
            empty_tiles += 1
    feature_vector.append(empty_tiles)

    bombs = game_state['bombs']
    bomb_nearby = any(abs(r - row) + abs(c - col) == 1 for (r, c), _ in bombs)
    feature_vector.append(int(bomb_nearby))

    coins = game_state['coins']
    coin_nearby = any(abs(r - row) + abs(c - col) == 1 for r, c in coins)
    feature_vector.append(int(coin_nearby))

    agent_score = game_state['self'][1]
    feature_vector.append(agent_score)

    feature_vector = np.array(feature_vector, dtype=np.float32)

    return feature_vector
