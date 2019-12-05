import copy
import numpy as np

class Maze():
    def __init__(self, x_length = 10, y_length = 10):
        self.x_length = x_length
        self.y_length = y_length

        self.ACTION_LT = 0
        self.ACTION_RT = 1

        self.ACTION_1D = [self.ACTION_LT, self.ACTION_RT]


    def make_1D(self):
        return np.ones((1, self.x_length))

    def square_maze(self):
        return np.ones((self.y_length, self.x_length))

    def make_barrier_maze_square(self, square_maze, barrier_length, barrier_x_position, \
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
    
    def step_1D(self, state, action):
        i, j = state
        if action == self.ACTION_LT:
            next_state = [i, max(j - 1, 0)]
        elif action == self.ACTION_RT:
            next_state = [i, min(j + 1, self.x_length - 1)]
        else:
            assert False
        return next_state
