# refactoring https://github.com/nicoring/hippocampus-predictive-map for me

# load libray
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mazemaking as mm
import sr

# maze env

MAZE_LENGTH = 100

square_maze = mm.square_maze(MAZE_LENGTH)

# put barrier
B_LENGTH = 60
B_THICKNESS = 10
B_X_POSTION = 20
B_Y_POSTION = 60
barriers = [[x, y] for x in range(B_Y_POSTION, B_Y_POSTION + B_THICKNESS) \
     for y in range(B_X_POSTION, B_X_POSTION + B_LENGTH)]

square_with_barrier = mm.make_barrier_maze_square(square_maze, \
    B_LENGTH, B_X_POSTION, B_Y_POSTION, B_THICKNESS)

SR_CELL = int(((B_Y_POSTION + B_THICKNESS) * MAZE_LENGTH) + \
    ((B_X_POSTION + B_LENGTH) / 2))

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

# run SR model

# prepare for SR matrix
all_states = [[x, y] for x in range(square_with_barrier.shape[0]) \
    for y in range(square_with_barrier.shape[1])]

sr_matrix = np.eye(len(all_states), dtype=np.float)

# history_sr_matrix = []

# history_exp_idx = 0
# 2D maze에서 history를 보는 것은 힘든 것 같다.

for i in tqdm(range(20)):
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
    plt.savefig('./images/sr_SQUARE_' + str(MAZE_LENGTH) + '.png')
    plt.close()

def figure_sr_point():
    fig, ax = plt.subplots() 
    ax.imshow(sr_point_with_barrier, cmap='gray', interpolation='nearest')
    '''
    for idx in range(sr_matrix.shape[0]):
        for jdx in range(sr_matrix.shape[1]):
            text = ax.text(jdx, idx,round(sr_matrix[idx, jdx],2), \
                ha = "center", va = "center", color = "b")
    '''        
    fig.tight_layout()
    plt.savefig('./images/sr_SQUARE_point' + str(MAZE_LENGTH) + '.png')
    plt.close()


'''
def figure_history():

    #원래 논문의 figure 2C에 해당됨.

    for_legend = []
    for idx in range(len(history_sr_matrix)):
        plt.plot(history_sr_matrix[idx])
        if idx == 0:
            for_legend.append("init")
        else:
            for_legend.append("10^"+str(idx-1))
    plt.legend(for_legend, loc = 'upper left')
    plt.savefig('./images/sr_histroy_' + str(MAZE_LENGTH) + '.png')
'''

if __name__=='__main__':
    figure_matrix()
    figure_sr_point()
 #   figure_history()

