# refactoring https://github.com/nicoring/hippocampus-predictive-map for me

# load libray
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mazemaking as mm
import sr

# maze env

MAZE_X_LENGTH = 40
MAZE_Y_LENGTH = 40


# put barrier
B_LENGTH = 20
B_THICKNESS = 1
B_X_POSTION = 10
B_Y_POSTION = 19


maze = mm.Maze(x_length = MAZE_X_LENGTH, y_length = MAZE_Y_LENGTH, \
    b_length = B_LENGTH, b_thickness = B_THICKNESS, b_x_position = B_X_POSTION, \
        b_y_position = B_Y_POSTION)

square_with_barrier = maze.make_barrier_maze_square()

SR_CELL = int(((B_Y_POSTION + B_THICKNESS + 1) * MAZE_Y_LENGTH) + \
    ((MAZE_X_LENGTH / 2) - 1))
RT_SR_CELL = int(SR_CELL - ((MAZE_X_LENGTH / 2) -1))
LT_SR_CELL = int(SR_CELL + (MAZE_X_LENGTH / 2)) 

# action on 2D maze
ACTIONS = [maze.ACTION_LT, maze.ACTION_RT, maze.ACTION_UP, maze.ACTION_DW]

# state information
START = [0, 0]
END = [MAZE_Y_LENGTH - 1, MAZE_X_LENGTH - 1]

# hyperparameter for updating SR matrix
alpha = 0.1
gamma = 0.99 # original parameter 0.13


# action policy
def choose_action(state):
    return np.random.randint(4)

# run SR model

# prepare for SR matrix
all_states = [[x, y] for x in range(square_with_barrier.shape[0]) \
    for y in range(square_with_barrier.shape[1])]

sr_matrix = np.eye(len(all_states), dtype=np.float)

history_sr_matrix = []

history_exp_idx = 0

for i in tqdm(range(1001)):
    state = START
    if i == 0: 
        history_sr_matrix.append(copy.deepcopy(sr_matrix[RT_SR_CELL:LT_SR_CELL,\
            SR_CELL]))
    
    while state != END:
        action = choose_action(state)
        next_state = maze.step_2D(state, action)
        sr_matrix = sr.update_SR_matrix(state, next_state, sr_matrix, \
             all_states, alpha = alpha, gamma = gamma)

        state = next_state
    if i == (10 ** history_exp_idx):
        history_sr_matrix.append(copy.deepcopy(sr_matrix[RT_SR_CELL:LT_SR_CELL,\
            SR_CELL]))
        history_exp_idx += 1


sr_point = copy.deepcopy(sr_matrix[:, SR_CELL])
sr_point = sr_point.reshape(square_with_barrier.shape)

sr_point_with_barrier = maze.barrier_for_sr_plot(sr_point)   
 
def figure2c_2D():
    fig, ax = plt.subplots() 
    ax.imshow(sr_point_with_barrier, cmap='viridis', interpolation='nearest')
    fig.tight_layout()
    plt.savefig('./images/figure2c_2D_online.png')
    plt.close()


def figure_history():
    for_legend = []
    for idx in range(len(history_sr_matrix)):
        plt.plot(history_sr_matrix[idx])
        if idx == 0:
            for_legend.append("init")
        else:
            for_legend.append("10^"+str(idx-1))
    plt.legend(for_legend, loc = 'upper left')
    plt.savefig('./images/figure2c_2D_online_history.png')



if __name__=='__main__':
    figure2c_2D()
    figure_history()

