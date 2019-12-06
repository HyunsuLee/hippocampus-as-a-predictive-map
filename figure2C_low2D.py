# analytic compute SR matrix from Trasitional matrix.
# intended to replicate figure 2C, 2D maze result

# load library
import matplotlib.pyplot as plt
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

SR_POINT = int(((B_Y_POSTION + B_THICKNESS + 1) * MAZE_Y_LENGTH) + \
    ((MAZE_X_LENGTH / 2) - 1))

gamma = 0.99

maze = mm.Maze(x_length = MAZE_X_LENGTH, y_length = MAZE_Y_LENGTH, \
    b_length = B_LENGTH, b_thickness = B_THICKNESS, b_x_position = B_X_POSTION, \
        b_y_position = B_Y_POSTION)

sqmaze = maze.make_barrier_maze_square()

ACTIONS = [maze.ACTION_LT, maze.ACTION_RT, maze.ACTION_UP, maze.ACTION_DW]

random_trans_mat = sr.transition_matrix(sqmaze, maze.step_2D, ACTION = ACTIONS)

sr_matrix = sr.analytic_M(random_trans_mat, gamma = gamma)

sr_point = sr_matrix[:, SR_POINT].reshape(sqmaze.shape)
sr_point = maze.barrier_for_sr_plot(sr_point)

def figure2c_2D():
    fig, ax = plt.subplots()
    ax.imshow(sr_point, cmap = 'viridis', interpolation = 'nearest')
    fig.tight_layout()
    plt.savefig('./images/figure2c_2D_analytic_99withBarrier.png')
    plt.close

if __name__ == '__main__':
    figure2c_2D()