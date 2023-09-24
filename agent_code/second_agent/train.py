from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
import numpy as np
from .tool import *
from .params import *

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

q_table = np.zeros((17 * 17, 6))  # Initialize with zeros

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    tr = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))
    log_debug(self, "FEATURE: " + str(state_to_features(old_game_state)))
    log_debug(self, reward_from_events(self, events))
    self.transitions.append(tr)

    # Update Q-table
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    update_model(self, old_state, new_state, reward_from_events(self, events), ACTION_INDEX[self_action])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    end_up_update_model(self, state_to_features(last_game_state), reward_from_events(self, events), ACTION_INDEX[last_action],)

    # q_table[state_to_features(last_game_state)] = reward_from_events(self, events)
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -50,
        e.COIN_COLLECTED: 5,
        e.CRATE_DESTROYED: 5,
        e.BOMB_DROPPED: 1,
        e.MOVED_LEFT: .2,
        e.MOVED_RIGHT: .2,
        e.MOVED_UP: .2,
        e.MOVED_DOWN: .2,
        e.INVALID_ACTION: 0,
        PLACEHOLDER_EVENT: 0  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_model(self, old_state, new_state, reward, action_index: int):
    Q_table = self.model
    q_prv_index = tuple(list(old_state) + [action_index])
    Q_table[q_prv_index] = (1 - learning_rate) * Q_table[q_prv_index] + learning_rate * (reward + discount_factor * np.max(Q_table[tuple(list(new_state))]))
    self.model = Q_table

def end_up_update_model(self, old_state, reward, action_index: int):
    Q_table = self.model
    q_prv_index = tuple(list(old_state) + [action_index])
    Q_table[q_prv_index] = (1 - learning_rate) * Q_table[q_prv_index] + learning_rate * reward
    self.model = Q_table
