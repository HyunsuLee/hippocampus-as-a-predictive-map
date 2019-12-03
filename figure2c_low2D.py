# refactoring https://github.com/nicoring/hippocampus-predictive-map for me

# load libray
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mazemaking as mm
import sr

# maze env

MAZE_LENGTH = 40

square_maze = mm.square_maze(MAZE_LENGTH)

# put barrier
B_LENGTH = 20
B_THICKNESS = 1
B_X_POSTION = 10
B_Y_POSTION = 19 
barriers = [[x, y] for x in range(B_Y_POSTION, B_Y_POSTION + B_THICKNESS) \
     for y in range(B_X_POSTION, B_X_POSTION + B_LENGTH)]

square_with_barrier = mm.make_barrier_maze_square(square_maze, \
    B_LENGTH, B_X_POSTION, B_Y_POSTION, B_THICKNESS)

SR_CELL = int(((B_Y_POSTION + B_THICKNESS + 1) * MAZE_LENGTH) + \
    ((MAZE_LENGTH / 2) - 1))

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
gamma = 0.13

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
def choose_action(state):
    return np.random.randint(4)

# run SR model

# prepare for SR matrix
all_states = [[x, y] for x in range(square_with_barrier.shape[0]) \
    for y in range(square_with_barrier.shape[1])]

sr_matrix = np.eye(len(all_states), dtype=np.float)

for i in tqdm(range(1000)):
    state = START
    
    while state != END:
        action = choose_action(state)
        next_state, _ = step(state, action)
        sr_matrix = sr.update_SR_matrix(state, next_state, sr_matrix, \
             all_states, alpha = alpha, gamma = gamma)

        state = next_state


sr_point = copy.deepcopy(sr_matrix[:, SR_CELL])
sr_point = sr_point.reshape(square_with_barrier.shape)

sr_point_with_barrier = mm.make_barrier_maze_square(sr_point, \
    B_LENGTH, B_X_POSTION, B_Y_POSTION, B_THICKNESS, for_sr = True)   
 
def figure2c_2D():
    fig, ax = plt.subplots() 
    ax.imshow(sr_point_with_barrier, cmap='viridis', interpolation='nearest')
    fig.tight_layout()
    plt.savefig('./images/figure2c_2D.png')
    plt.close()



if __name__=='__main__':
    figure2c_2D()

