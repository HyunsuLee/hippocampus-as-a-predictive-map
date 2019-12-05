# analytic compute SR matrix from Trasitional matrix.
# intended to replicate figure 2C, 2D maze result

# load library
import matplotlib.pyplot as plt
import mazemaking as mm
import sr

# maze env
MAZE_X_LENGTH = 40 
MAZE_Y_LENGTH = 40

SR_POINT = 20

gamma = 0.99

maze = mm.Maze(x_length = MAZE_X_LENGTH, y_length = MAZE_Y_LENGTH)
sqmaze = maze.square_maze()

ACTIONS = [maze.ACTION_LT, maze.ACTION_RT, maze.ACTION_UP, maze.ACTION_DW]

random_trans_mat = sr.transition_matrix(sqmaze, maze.step_2D, ACTION = ACTIONS)

sr_matrix = sr.analytic_M(random_trans_mat, gamma = gamma)

sr_point = sr_matrix[:, SR_POINT]
sr_point = sr_point.reshape(sqmaze.shape)

def figure2c_2D():
    fig, ax = plt.subplots()
    ax.imshow(sr_point, cmap = 'viridis', interpolation = 'nearest')
    fig.tight_layout()
    plt.savefig('./images/figure2c_2D_analytic_99.png')
    plt.close

if __name__ == '__main__':
    figure2c_2D()
