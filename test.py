import mazemaking as mm
import sr

# maze env

maze = mm.Maze(x_length = 3, y_length=10)
dmaze = maze.make_1D()

random_trans_mat= sr.transition_matrix(dmaze, maze.step_1D, policy=True)
print(random_trans_mat)
print(sr.analytic_M(random_trans_mat))
