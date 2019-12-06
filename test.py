import numpy as np

test_np = np.array([[1, 0,1], [1, 0, 1], [0,0,0], [1,0,1]])

def remove_zeros(test_np):
    zero_row_idx = np.argwhere(np.all(test_np == 0, axis = 1))
    zero_col_idx = np.argwhere(np.all(test_np == 0, axis = 0))
    test_np = np.delete(test_np, zero_row_idx, axis=0)
    test_np = np.delete(test_np, zero_col_idx, axis=1)
    return test_np

print(remove_zeros(test_np))
