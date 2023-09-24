import os
import pickle
import random
from .tool import log_debug, log_info
from .feature_utils import *
from .params import *

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

    if self.train or not os.path.isfile("my-saved-model.pt"):
        if keep_model and os.path.isfile("my-saved-model.pt"):
            self.logger.info("Training mode, keep old model data, loading model from saved state.")
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
        else:
            self.logger.info("Setting up model from scratch.")
            self.model = np.zeros(feature_size)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    # If not found agent_q_table.npy file, use random action.
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    features: np.ndarray = state_to_features(game_state)
    self.logger.debug("Querying model for action.")

    if (not hasattr(self, "model")):
        self.model = np.zeros(tuple(np.shape(state_to_features(game_state))))

    log_debug(self, "feature:" + str(tuple(features)))
    action_index = np.argmax(self.model[tuple(features)])
    log_debug(self, "action-index:" + str(action_index))
    # log_debug(self, "self.model:" + str(self.model))
    return ACTIONS[action_index]


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
    self_info = game_state['self']
    other_agents = game_state['others']
    bombs = game_state['bombs']
    coins = game_state['coins']
    user_input = game_state['user_input']

    player_pos = self_info[3]
    others_pos = [other[3] for other in other_agents]


    features_vector = []

    # Coin collect
    features_vector.append(coin_detector(field, coins, player_pos))
    features_vector.append(np.array(16 if cal_nearest_obj_distance(field, coins, player_pos) >= 16 else cal_nearest_obj_distance(field, coins, player_pos)).reshape(1))

    # Danger avoidance
    features_vector.append(is_danger(field, player_pos, bombs))

    # Crates detection
    features_vector.append(is_crates_around(field, player_pos))

    # Enemy detection
    features_vector.append(is_enemy_nearby(player_pos, others_pos))

    # Self bomb-left status
    features_vector.append([self_info[2]])

    return np.concatenate(features_vector).astype(int)


