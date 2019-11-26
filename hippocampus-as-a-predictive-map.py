# refactoring https://github.com/nicoring/hippocampus-predictive-map for me

# load libray
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# simple 1D maze with prefered direction or random policy

# maze env

def make_1D(length):
    return np.ones((1, length))

MAZE_LENGTH = 10

maze = make_1D(MAZE_LENGTH)

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

# for matrix M
def state_to_idx(state, s_prime, next_state, all_states):
    for idx in range(len(all_states)):
        if all_states[idx] == state:
            idx_state = idx
    for idx in range(len(all_states)):
        if all_states[idx] == s_prime:
            idx_s_prime = idx    
    for idx in range(len(all_states)):
        if all_states[idx] == next_state:
            idx_next_state = idx        
    
    return idx_state, idx_s_prime, idx_next_state


def update_SR_matrix(state, s_prime, next_state, I, next_sr_matrix,\
    all_states, alpha = alpha, gamma = gamma):
    idx_state, idx_s_prime, idx_next_state = \
        state_to_idx(state, s_prime, next_state, all_states)
    M_element = next_sr_matrix[idx_state, idx_s_prime]
    M_next_state_element = next_sr_matrix[idx_next_state, idx_s_prime]
    new_M_element = M_element + alpha * (I + gamma * M_next_state_element - \
        M_element)
    next_sr_matrix[idx_state, idx_s_prime] = new_M_element
    if idx_state == idx_s_prime:
        next_sr_matrix[idx_state, idx_s_prime] = 1
    return next_sr_matrix  



    

# r = np.array([[0,0,0,0,1]])

# prepare for SR matrix
values = np.zeros(maze.shape, dtype=np.float)
visited = np.zeros(maze.shape)
all_states = [[x, y] for x in range(maze.shape[0]) for y in range(maze.shape[1])]
# 여기서 state란 좌표를 의미한다. 꼭 좌표로 받아야 하나?
# all_states = [np.array(s) for s in it.product(range(maze.shape[0], \
#    range(maze.shapes[1])))]

n_states = maze.shape[0] * maze.shape[1]
sr_matrix = np.eye(len(all_states), dtype=np.float)


s = None



next_sr_matrix = sr_matrix.copy()

for i in tqdm(range(1000)):
    state = START
    while state != END:
        action = choose_action(state, prefered=True)
        next_state, _ = step(state, action)
        visited[next_state[0], next_state[1]] += 1.0
        for s_prime in all_states:
            I = 1 if state == s_prime else 0
            next_sr_matrix = update_SR_matrix(state, s_prime, next_state, I, \
                next_sr_matrix, all_states)

        state = next_state  
def figure():  
    fig, ax = plt.subplots() 
    im = ax.imshow(next_sr_matrix, cmap='gray', interpolation='nearest')
    for idx in range(next_sr_matrix.shape[0]):
        for jdx in range(next_sr_matrix.shape[1]):
            text = ax.text(jdx, idx,round(next_sr_matrix[idx, jdx],2), \
                ha = "center", va = "center", color = "b")
            
    fig.tight_layout()
    plt.savefig('./images/sr_matrix_10.png')
    plt.close()

if __name__=='__main__':
    figure()


