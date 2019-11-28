# refactoring https://github.com/nicoring/hippocampus-predictive-map for me

# load libray
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mazemaking as mm
import sr

# maze env

MAZE_LENGTH = 10

square_maze = mm.square_maze(MAZE_LENGTH)

# put barrier
B_LENGTH = 5
B_X_POSTION = 2
B_Y_POSTION = 4
barriers = [[B_Y_POSTION, y] for y in range(B_X_POSTION, B_X_POSTION + B_LENGTH)]

square_with_barrier = mm.make_barrier_maze_square(square_maze, \
    B_LENGTH, B_X_POSTION, B_Y_POSTION)

# action on 2D maze
ACTION_UP = 0
ACTION_RT = 1
ACTION_DW = 2
ACTION_LT = 3

ACTIONS = [ACTION_UP, ACTION_RT, ACTION_DW, ACTION_LT]

# state information
START = [0, 0]
END = [MAZE_LENGTH - 1, MAZE_LENGTH - 1]

# hyperparameter for updating SR matrix
alpha = 0.1
gamma = 0.9

# make actions and receive reward
def step(state, action):
    i, j = state
    if action == ACTION_LT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RT:
        next_state = [i, min(j + 1, MAZE_LENGTH - 1)]
    elif action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_DW:
        next_state = [min(i + 1, MAZE_LENGTH -1), j]
    else:
        assert False
    if next_state in barriers:
        next_state = state
    
    reward = 0

    return next_state, reward

# action policy
def choose_action(state, policy = False):
    '''
    TODO 나중에 Qlearning으로 변환할 경우 q value, epsilon-greedy로 수정할 것.
    지금은 그냥 fixed random policy임. 그래서 state는 현재로서는 쓰이지 않는다.
    '''
    if policy == True:
        pass
    elif policy == False:
        action = np.random.randint(4)
    else:
        assert False
    return action

print(step([0,0], choose_action([0,0])))
