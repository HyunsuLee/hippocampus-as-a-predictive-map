import copy
import numpy as np

class Maze():
    def __init__(self, x_length = 10, y_length = 10, b_length = 20, \
        b_thickness = 1, b_x_position = 10, b_y_position = 19):
        self.x_length = x_length
        self.y_length = y_length

        self.ACTION_LT = 0
        self.ACTION_RT = 1
        self.ACTION_UP = 2
        self.ACTION_DW = 3

        self.b_length = b_length
        self.b_thickness = b_thickness
        self.b_x_position = b_x_position
        self.b_y_position = b_y_position 

    def make_1D(self):
        return np.ones((1, self.x_length))

    def square_maze(self):
        return np.ones((self.y_length, self.x_length))

    def make_barrier_maze_square(self):
        '''
        make horizontal barrier in the square maze.
        '''
        square_maze = np.ones((self.y_length, self.x_length))
        if square_maze.shape[0] < self.b_length:
            print('barrier cannot exceed the size of maze')
            raise ValueError
        elif square_maze.shape[1] < (self.b_x_position + self.b_length):
            print('cannot put barrier outside of the maze')
            raise ValueError
        elif square_maze.shape[0] < self.b_y_position:
            print('cannot put barrier outside of the maze')
            raise ValueError
        else:
            pass
        new_maze = copy.deepcopy(square_maze)

        new_maze[self.b_y_position:self.b_y_position + self.b_thickness, \
            self.b_x_position:self.b_x_position + self.b_length] = 0
                
        return new_maze
    
    def barrier_for_sr_plot(self, sr_heatmap):
        sr_heatmap[self.b_y_position:self.b_y_position + self.b_thickness, \
            self.b_x_position:self.b_x_position + self.b_length] = 1
        return sr_heatmap
    
    def step_1D(self, state, action):
        i, j = state
        if action == self.ACTION_LT:
            next_state = [i, max(j - 1, 0)]
        elif action == self.ACTION_RT:
            next_state = [i, min(j + 1, self.x_length - 1)]
        else:
            assert False
        return next_state

    def step_2D(self, state, action):
        i, j = state
        if action == self.ACTION_LT:
            next_state = [i, max(j - 1, 0)]
        elif action == self.ACTION_RT:
            next_state = [i, min(j + 1, self.x_length -1)]
        elif action == self.ACTION_UP:
            next_state = [max(i - 1, 0), j]
        elif action == self.ACTION_DW:
            next_state = [min(i + 1, self.y_length -1), j]
        else:
            assert False
        return next_state

