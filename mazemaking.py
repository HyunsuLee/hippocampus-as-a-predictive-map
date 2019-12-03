import copy
import numpy as np

def make_1D(length):
    return np.ones((1, length))

def square_maze(size):
    return np.ones((size, size))

def make_barrier_maze_square(square_maze, barrier_length, barrier_x_position, \
    barrier_y_position, barrier_thickness = 1, for_sr = False):
    '''
    make horizontal barrier in the square maze.
    '''
    if square_maze.shape[0] < barrier_length:
        print('barrier cannot exceed the size of maze')
        raise ValueError
    elif square_maze.shape[1] < (barrier_x_position + barrier_length):
        print('cannot put barrier outside of the maze')
        raise ValueError
    elif square_maze.shape[0] < barrier_y_position:
        print('cannot put barrier outside of the maze')
        raise ValueError
    else:
        pass
    new_maze = copy.deepcopy(square_maze)

    new_maze[barrier_y_position:barrier_y_position + barrier_thickness, \
        barrier_x_position:barrier_x_position + barrier_length] = 0 \
            if for_sr == False else 1

    return new_maze
