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
        self.logger.info("Q-table file not found.")



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

    # If not found agent_q_table.npy file
    if (not hasattr(self, 'q_table')):
        self.logger.info("q-table file not found, use random strategy instead.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # If it has agent_q_table.npy file
    # self.logger.info(self.q_table)
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

    round_num = game_state['round']
    step_num = game_state['step']
    field = game_state['field']
    self_info = game_state['self']
    other_agents = game_state['others']
    bombs = game_state['bombs']
    coins = game_state['coins']
    user_input = game_state['user_input']

    field_vector = field.flatten()

    self_position = self_info[3]
    self_vector = np.array(self_position)

    other_agents_positions = [other[3] for other in other_agents]
    other_agents_vector = np.array(other_agents_positions).flatten()

    # bombs_positions = [bomb[2] for bomb in bombs]
    # bombs_vector = np.array(bombs_positions).flatten()
    bombs_vector = np.array([])

    # coins_positions = [coin.get_state() for coin in coins]
    # coins_vector = np.array(coins_positions).flatten()
    coins_vector = np.array([])

    user_input_vector = np.array(user_input) if user_input is not None else np.array([])

    feature_vector = np.concatenate(
        [field_vector, self_vector, other_agents_vector, bombs_vector, coins_vector, user_input_vector])

    return feature_vector

