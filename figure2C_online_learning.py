# refactoring https://github.com/nicoring/hippocampus-predictive-map
# this code intended for replicatig the figure 2C.

# load libray
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mazemaking as mm
import sr

# simple 1D maze with agent going right.

# maze env

MAZE_LENGTH = 50 # p.8 of supplement, 
# the paper said that 500 states were used for 1D maze, it seems to be typo
# 50 states shows similar results with the figure 2C.
SR_POINT = int(MAZE_LENGTH * 0.75)

maze = mm.Maze(x_length = MAZE_LENGTH)
dmaze = maze.make_1D()

# action on 1D maze
ACTION_LT = 0
ACTION_RT = 1
ACTIONS = [ACTION_LT, ACTION_RT]

# state information
START = [0, 0]
END = [0, MAZE_LENGTH - 1]

# hyperparameter for updating SR matrix
alpha = 0.1
gamma = 0.84 # p.8 of supplement
# the paper said that 0.084 gamma was used for 1D maze,
# 0.84 gamma shows similar results. With smaller gamma, the sharper SR rate shows.



# prepare for SR matrix
all_states = [[x, y] for x in range(dmaze.shape[0]) for y in range(dmaze.shape[1])]

sr_matrix = np.eye(len(all_states), dtype=np.float)

history_sr_matrix = []

history_exp_idx = 0

for i in tqdm(range(1001)):
    state = START
    if i == 0: 
        history_sr_matrix.append(copy.deepcopy(sr_matrix[:, SR_POINT]))
    
    while state != END:
        action = ACTION_RT 
        next_state = maze.step_1D(state, action)
        sr_matrix = sr.update_SR_matrix(state, next_state, sr_matrix, \
             all_states, alpha = alpha, gamma = gamma)

        state = next_state

    if i == (10 ** history_exp_idx):
       history_sr_matrix.append(copy.deepcopy(sr_matrix[:, SR_POINT]))
       history_exp_idx += 1
    
 
def figure_matrix():  
    fig, ax = plt.subplots() 
    ax.imshow(sr_matrix, cmap='gray', interpolation='nearest')
    '''
    for idx in range(sr_matrix.shape[0]):
        for jdx in range(sr_matrix.shape[1]):
            text = ax.text(jdx, idx,round(sr_matrix[idx, jdx],2), \
                ha = "center", va = "center", color = "b")
    this code intended for text annotation for probability. 
    however too many cells are drawn to annotate each probability.
    '''        
    fig.tight_layout()
    plt.savefig('./images/figure2C_1D_online_SRmatrix.png')
    plt.close()

def figure_history():
    '''
    draw figure 2C
    '''
    for_legend = []
    for idx in range(len(history_sr_matrix)):
        plt.plot(history_sr_matrix[idx])
        if idx == 0:
            for_legend.append("init")
        else:
            for_legend.append("10^"+str(idx-1))
    plt.legend(for_legend, loc = 'upper left')
    # plt.savefig('./images/sr_histroy_' + str(MAZE_LENGTH) + '.png')
    plt.savefig('./images/figure2c_online.png')

if __name__=='__main__':
    figure_matrix()
    figure_history()


