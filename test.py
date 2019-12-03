import mazemaking as mm

# maze env

MAZE_LENGTH = 10

square_maze = mm.square_maze(MAZE_LENGTH)

# put barrier
B_LENGTH = 6
B_THICKNESS = 1
B_X_POSTION = 2
B_Y_POSTION = 6
barriers = [[B_Y_POSTION, y] for y in range(B_X_POSTION, B_X_POSTION + B_LENGTH)]

square_with_barrier = mm.make_barrier_maze_square(square_maze, \
    B_LENGTH, B_X_POSTION, B_Y_POSTION, B_THICKNESS)

barriers = [[x, y] for x in range(B_Y_POSTION, B_Y_POSTION + B_THICKNESS) \
     for y in range(B_X_POSTION, B_X_POSTION + B_LENGTH)]

SR_CELL = int(((B_Y_POSTION + B_THICKNESS) * MAZE_LENGTH) + \
    ((B_X_POSTION + B_LENGTH) / 2))
print(SR_CELL)