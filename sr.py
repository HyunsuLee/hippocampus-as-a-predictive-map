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
    #sr_matrix[idx_state, idx_state] = 1
    return sr_matrix  