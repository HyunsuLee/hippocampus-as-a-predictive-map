# refactoring https://github.com/nicoring/hippocampus-predictive-map for me

# load libray
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mazemaking as mm
import sr

# simple 1D maze with prefered direction or random policy

# maze env

MAZE_LENGTH = 10

maze = mm.make_1D(MAZE_LENGTH)

# action on 1D maze
ACTION_LT = 0
ACTION_RT = 1
ACTIONS = [ACTION_LT, ACTION_RT]

# state information
START = [0, 0]
END = [0, MAZE_LENGTH - 1]

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
    else:
        assert False
    
    reward = 0

    return next_state, reward

# action policy
def choose_action(state, prefered = True):
    '''
    TODO 나중에 Qlearning으로 변환할 경우 q value, epsilon-greedy로 수정할 것.
    지금은 그냥 fixed policy임. 
    '''
    if prefered == True:
        action = np.random.binomial(1, 0.8)
    elif prefered == False:
        action = np.random.binomial(1, 0.5)
    else:
        assert False
    return action



# prepare for SR matrix
all_states = [[x, y] for x in range(maze.shape[0]) for y in range(maze.shape[1])]

sr_matrix = np.eye(len(all_states), dtype=np.float)

history_sr_matrix = []

history_exp_idx = 0

for i in tqdm(range(1001)):
    state = START
    if i == 0: 
        history_sr_matrix.append(copy.deepcopy(sr_matrix[:, len(all_states)-1]))
    
    while state != END:
        action = choose_action(state, prefered=True)
        next_state, _ = step(state, action)
        sr_matrix = sr.update_SR_matrix(state, next_state, sr_matrix, \
             all_states, alpha = alpha, gamma = gamma)

        state = next_state

    if i == (10 ** history_exp_idx):
       history_sr_matrix.append(copy.deepcopy(sr_matrix[:, len(all_states)-1]))
       history_exp_idx += 1
    
 
def figure_matrix():  
    fig, ax = plt.subplots() 
    ax.imshow(sr_matrix, cmap='gray', interpolation='nearest')
    '''
    for idx in range(sr_matrix.shape[0]):
        for jdx in range(sr_matrix.shape[1]):
            text = ax.text(jdx, idx,round(sr_matrix[idx, jdx],2), \
                ha = "center", va = "center", color = "b")
    '''        
    fig.tight_layout()
    plt.savefig('./images/sr_matrix_' + str(MAZE_LENGTH) + '.png')
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
    plt.savefig('./images/sr_histroy_' + str(MAZE_LENGTH) + '.png')


if __name__=='__main__':
    figure_matrix()
    figure_history()


