import numpy as np

# for matrix M
def state_to_idx(state, next_state, all_states):
    for idx in range(len(all_states)):
        if all_states[idx] == state:
            idx_state = idx  
    for idx in range(len(all_states)):
        if all_states[idx] == next_state:
            idx_next_state = idx        
    return idx_state, idx_next_state


def update_SR_matrix(state, next_state, sr_matrix, all_states, alpha = 0.1, \
    gamma = 0.9):
    idx_state, idx_next_state = state_to_idx(state, next_state, all_states)
    I = np.zeros(sr_matrix[0, :].shape)
    I[idx_state] = 1
    M_state_V = sr_matrix[idx_state, :]
    M_next_state_V = sr_matrix[idx_next_state, :]
    sr_matrix[idx_state, :] = M_state_V + alpha * (I + \
        gamma * M_next_state_V - M_state_V)
    return sr_matrix

# analytic compute SR with transition matrix
def transition_matrix(maze, step, ACTION=[0,1], policy = "random"):
    all_states = [[x, y] for x in range(maze.shape[0]) \
        for y in range(maze.shape[1])]
    adj_matrix = np.zeros((len(all_states), len(all_states)))

    for state in all_states:
        i, j = state
        if maze[i, j] == 0:
            pass
        else:
            for action in ACTION:
                next_state = step(state, action)  
                i_prime, j_prime = next_state
                idx_state, idx_next_state = state_to_idx(state, next_state, \
                    all_states)
                if maze[i_prime, j_prime] == 0:
                    pass
                elif policy == "random":
                    adj_matrix[idx_state, idx_next_state] = 1
                elif policy == "RT":
                    if action == 1:
                        adj_matrix[idx_state, idx_next_state] = 1
                    elif action == 0:
                        pass
    end_sr = adj_matrix.shape[0]
    adj_matrix[end_sr-1, end_sr-1] = 0
    if policy == "random":
        return adj_matrix/sum(adj_matrix)
    elif policy == "RT":
        return adj_matrix

def analytic_M(transition_matrix, gamma = 0.9):
    identity_matrix = np.eye(transition_matrix.shape[0])
    sr_matrix = np.linalg.inv(identity_matrix - (gamma * transition_matrix))
    return sr_matrix

