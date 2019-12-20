# refactoring https://github.com/nicoring/hippocampus-predictive-map
# this code intended for replicatig the figure 2C.
# analytic compute SR matrix from Transitional matrix.

# load libray
import matplotlib.pyplot as plt
import mazemaking as mm
import sr

# maze env

MAZE_LENGTH = 50
SR_POINT = int(MAZE_LENGTH * 0.75)

gamma = 0.84

maze = mm.Maze(x_length = MAZE_LENGTH)
dmaze = maze.make_1D()

random_trans_mat, zero_row_idx, zero_col_idx= sr.transition_matrix(dmaze, maze.step_1D, policy="RT")

sr_matrix = sr.analytic_M(random_trans_mat, zero_row_idx, zero_col_idx, gamma=gamma)

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
    plt.savefig('./images/figure2C_analytic_sr_matrix.png')
    plt.close()


def figure_plot():
    plt.plot(sr_matrix[:, SR_POINT])
    plt.savefig('./images/figure2C_analytic_plot.png')


if __name__=='__main__':
    figure_matrix()
    figure_plot()
