import numpy as np
from queue import Queue
from settings import *

feature_size = [2,2,2,2, # coin
                17, # coin distance
                4,4,4,4, # bomb danger predict
                2, # adjacent crates
                2, # enemy around
                2, # bomb left
                6] # num


# Coin collect
"""
Detect which side has coins available
"""
def coin_detector(field, coins, player_pos):
    if not coins:
        return np.zeros(4)

    x, y = player_pos
    distances = [cal_nearest_obj_distance(field, coins, (x + dx, y + dy)) if field[x + dx, y + dy] == 0 else 0 for
                 dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]]

    curr_dist = cal_nearest_obj_distance(field, coins, player_pos)
    to_coin = [1 if 0 < d < curr_dist else 0 for d in distances]

    return to_coin


def cal_nearest_obj_distance(field, objects, start_position):
    if not objects:
        return 0

    dimensions = np.shape(field)
    parents = np.full(dimensions, -1)
    queue = Queue()
    queue.put((start_position, 1))

    while not queue.empty():
        current_pos, current_dist = queue.get()

        if parents[current_pos] > -1:
            continue
        parents[current_pos] = 0

        if current_pos in objects:
            return current_dist

        x, y = current_pos
        neighbors = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
        for neighbor in neighbors:
            nx, ny = neighbor
            if 0 <= nx < dimensions[0] and 0 <= ny < dimensions[1] and field[nx, ny] == 0:
                queue.put((neighbor, current_dist + 1))
    return 0


# Bomb avoidance
def is_danger(field, player_pos, bombs):
    """
    -1 cannot reach
    0 safe
    1 danger
    2 will die in next turn

    """
    width, height = field.shape
    danger_field = get_danger_field(field, bombs)
    danger_predict = []
    x, y = player_pos
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        tx, ty = x + dx, y + dy
        neighbor_pos = (tx, ty)
        if tx < 0 or ty < 0 or tx >= width or ty >= height:
            danger_predict.append(0)
        elif field[neighbor_pos] != 0:
            danger_predict.append(0)
        elif danger_field[neighbor_pos] <= BOMB_POWER:
            danger_predict.append(2)
        elif danger_field[neighbor_pos] != 99:
            danger_predict.append(1)
        else:
            danger_predict.append(0)


    return danger_predict

def get_danger_field(field, bombs):
    width, height = field.shape
    danger_field = np.full((width, height), 99)

    for (x, y), countdown in bombs:
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            i = 1
            danger_field[x, y] = countdown
            while i <= BOMB_POWER and 0 <= x + i * dx < width and 0 <= y + i * dy < height and field[x + i * dx, y + i * dy] != -1:
                danger_field[x + i * dx, y + i * dy] = min(countdown, danger_field[x + i * dx, y + i * dy])
                i += 1

    return np.transpose(danger_field)

def _get_potential_explosion_area_for_single_bomb(field, bomb_pos):
    potential_explosion_area = []
    x, y = bomb_pos
    for i in range(1, BOMB_POWER):
        if (field[x, y+i] != -1):
            potential_explosion_area.append([x, y+i])
    for i in range(1, BOMB_POWER):
        if (field[x+i, y] != -1):
            potential_explosion_area.append([x+i, y])
    for i in range(1, BOMB_POWER):
        if (field[x, y-i] != -1):
            potential_explosion_area.append([x, y-i])
    for i in range(1, BOMB_POWER):
        if (field[x-i, y] != -1):
            potential_explosion_area.append([x-i, y])
    return potential_explosion_area


# crater destroyer
def is_crates_around(field, player_pos):
    (x, y) = player_pos
    if field[x+1,y] == 1 or field[x-1, y] == 1 or field[x, y+1] == 1 or field[x, y-1] == 1:
        return [1]
    else:
        return [0]



# Enemy attack
def is_enemy_nearby(player_pos, others_pos):
    (x, y) = player_pos
    for other_pos in others_pos:
        ox, oy = other_pos
        if ox == x+1 or ox == x-1 or oy == y+1 or oy == y-1:
            return [1]
    return [0]

