#!/bin/python
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018
#

'''
    1) Heuristic 1:
        --- To start with - we calculated the manhattan distance for each number with respect to it's goal state.
        --- Later we slightly modified it, for example:
            If '16' is at position (1,1) in the board manhattan calculates it as 6 where as it must be 2. So, corrected this logic.
        --- Figured that this underestimates certain states like:
            1 2 3 4
            5 6 7 8
            9 10 12 11
            13 14 15 16
            Although this produces manhattan distances as as just 2, this board is very difficult to solve.
        --- So in order to design a heuristic which neither overstimates nor underestimates, we came up with few board positions:
            which look easy but extremely difficult to solve (like above) and vice versa

    2) Heuristic 2:
        --- In this, we tried to check the surroundings of numbers in the board and sum up their differences. For example,
            Estimate for 6 in the above example:
                left_difference = 6-5 = 1
                right_difference = 6-7 = -1
                up_difference = 6-2/4 = 1
                down_difference = 6-10/4 = -1
                Summing up all of them = 0.. which is an ideal case for any number.
            For 1 in the above example:
                left_difference = 1-4/3 = -1
                right_difference = 1-2 = -1
                up_difference = 1-13/12 = -1
                down_difference = 1-5/4 = -1
            So, by adding these values for all the positions in the board, eventuallu we get the value as 0.

        --- Here also considered the cases like 4 3 2, where the left and right differences sums to 0, just like 2 3 4. So, updated the logic
            to handle such cases too.

        --- This seem to have correctly predicted the costs for board2, 4, 6, 8, 10 but underestimates the cost for board12.
            After investigating figured that the board12 4 2 3 and 9 11 12, also it produces the differences in just magnitude of 1 in
            this case, but the actual cost is much higher

    3) Heuristic 3:
        --- So, to tackle the above mentioned problem, made few additions to the heuristic 2 to handle such cases.'''

from Queue import PriorityQueue
from random import randrange, sample
import numpy as np
import sys
import string

# shift a specified row left (-1) or right (1)
def shift_row(state, row, dir):
    change_row = state[(row * 4):(row * 4 + 4)]
    return (state[:(row * 4)] + change_row[-dir:] + change_row[:-dir] + state[(row * 4 + 4):],
            ("L" if dir == -1 else "R") + str(row + 1))


# shift a specified col up (1) or down (-1)
def shift_col(state, col, dir):
    change_col = state[col::4]
    s = list(state)
    s[col::4] = change_col[-dir:] + change_col[:-dir]
    return (tuple(s), ("U" if dir == -1 else "D") + str(col + 1))


# pretty-print board state
def print_board(row):
    for j in range(0, 16, 4):
        print '%3d %3d %3d %3d' % (row[j:(j + 4)])


# return a list of possible successor states
def successors(state):
    return [shift_row(state, i, d) for i in range(0, 4) for d in (1, -1)] + [shift_col(state, i, d) for i in range(0, 4)
                                                                             for d in (1, -1)]


# just reverse the direction of a move name, i.e. U3 -> D3
def reverse_move(state):
    return state.translate(string.maketrans("UDLR", "DURL"))


# check if we've reached the goal
def is_goal(state):
    return sorted(state) == list(state)


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))

def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))

def heuristic_value3(state):
    goal = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    matrix = [list(state[4 * x:4 * x + 4]) for x in range(4)]

    indices = []
    goal = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    matrix = [list(state[4 * x:4 * x + 4]) for x in range(4)]
    for i in range(4):
        if not ((i * 4) + 1 in matrix[i] or (i * 4) + 2 in matrix[i] or (i * 4) + 3 in matrix[i] or (i * 4) + 4 in
                matrix[i]):
            hVal = hVal + 1
        elif (((i * 4) + 1 in matrix[i] and matrix[i][0] == (i * 4) + 1) or (i * 4) + 1 not in matrix[i]) and (
        (((i * 4) + 2 in matrix[i] and matrix[i][1] == (i * 4) + 2) or (i * 4) + 2 not in matrix[i])) and (
        (((i * 4) + 3 in matrix[i] and matrix[i][2] == (i * 4) + 3) or (i * 4) + 3 not in matrix[i])) and (
        (((i * 4) + 4 in matrix[i] and matrix[i][3] == (i * 4) + 4) or (i * 4) + 4 not in matrix[i])):
            continue
        else:
            hVal = hVal + 1

    for j in range(4):
        if not (j + 1 in [matrix[k][j] for k in range(4)] or j + 5 in [matrix[k][j] for k in range(4)] or j + 9 in [
            matrix[k][j] for k in range(4)] or j + 13 in [matrix[k][j] for k in range(4)]):
            hVal = hVal + 1
        elif ((j + 1 in [matrix[k][j] for k in range(4)] and matrix[0][j] == j + 1) or j + 1 not in [matrix[k][j] for k
                                                                                                     in range(4)]) and (
        ((j + 5 in [matrix[k][j] for k in range(4)] and matrix[1][j] == j + 5) or j + 5 not in [matrix[k][j] for k in
                                                                                                range(4)])) and ((
                (j + 9 in [matrix[k][j] for k in range(4)] and matrix[2][j] == j + 9) or j + 9 not in [matrix[k][j] for
                                                                                                       k in
                                                                                                       range(4)])) and (
        ((j + 13 in [matrix[k][j] for k in range(4)] and matrix[3][j] == j + 13) or j + 13 not in [matrix[k][j] for k in
                                                                                                   range(4)])):
            continue
        else:
            hVal = hVal + 1
    # print(hVal)
    for i in range(4):
        for j in range(3):
            row_cost = 0
            col_cost = 0
            if (matrix[i][j] in goal[i] and matrix[i][j + 1] in goal[i]):
                if matrix[i][j] - matrix[i][j + 1] == -2:
                    if abs(index_2d(matrix, matrix[i][j + 1] - 1)[0] - i) == 3:
                        row_cost = 1
                    else:
                        row_cost = abs(index_2d(matrix, matrix[i][j + 1] - 1)[0] - i)
                    if abs(index_2d(matrix, matrix[i][j + 1] - 1)[1] - j) == 3:
                        col_cost = 1
                    else:
                        col_cost = abs(index_2d(matrix, matrix[i][j + 1] - 1)[1] - j)
                    hVal = hVal + row_cost + col_cost
                    # hVal = hVal + abs(index_2d(matrix, matrix[i][j+1] - 1)[0] - i) + abs(index_2d(matrix, matrix[i][j+1] - 1)[1] - j)
                elif matrix[i][j] - matrix[i][j + 1] == 1:
                    hVal = hVal + 4
                elif matrix[i][j] - matrix[i][j + 1] == 2:
                    if abs(index_2d(matrix, matrix[i][j] - 1)[0] - i) == 3:
                        row_cost = 1
                    else:
                        row_cost = abs(index_2d(matrix, matrix[i][j] - 1)[0] - i)
                    if abs(index_2d(matrix, matrix[i][j] - 1)[1] - j - 1) == 3:
                        col_cost = 1
                    else:
                        col_cost = abs(index_2d(matrix, matrix[i][j] - 1)[1] - j - 1)
                    hVal = hVal + row_cost + col_cost
                    # hVal = hVal + abs(index_2d(matrix, matrix[i][j] - 1)[0] - i) + abs(index_2d(matrix, matrix[i][j] - 1)[1] - j - 1)

    for j in range(4):
        for i in range(3):
            if (matrix[i][j] in [goal[k][j] for k in range(4)] and matrix[i + 1][j] in [goal[k][j] for k in range(4)]):
                if matrix[i][j] - matrix[i + 1][j] == -2:
                    hVal = hVal + abs(index_2d(matrix, matrix[i + 1][j] - 1)[0] - i) + abs(
                        index_2d(matrix, matrix[i + 1][j] - 1)[1] - j)
                elif matrix[i + 1][j] - matrix[i][j] == 1:
                    hVal = hVal + 2
                elif matrix[i][j] - matrix[i + 1][j] == 2:
                    hVal = hVal + abs(index_2d(matrix, matrix[i][j] - 1)[0] - i - 1) + abs(
                        index_2d(matrix, matrix[i][j] - 1)[1] - j)
    return hVal

def heuristic_value2(state):
    matrix = [list(state[4 * x:4 * x + 4]) for x in range(4)]
    left = 0
    right = 0
    up = 0
    down = 0
    res = 0
    for i in range(4):
        for j in range(4):
            left = 0
            right = 0
            up = 0
            down = 0
            if i == 0:
                #hVal += matrix[i][j] - matrix[i][(j + 1)%4]
                up += abs(float(matrix[i][j] - matrix[i - 1][j]) / 12)
                down += float(matrix[i][j] - matrix[i+1][j])/4
                #hVal += matrix[i][j] - matrix[i][j-1]/3
            elif i == 3:
                up += abs(float(matrix[i][j] - matrix[i - 1][j]) / 4)
                down += float(matrix[i][j] - matrix[(i + 1) % 4][j]) / 12
            if j == 0:
                left += abs(float(matrix[i][j] - matrix[i][j-1]) / 3)
                right += matrix[i][j] - matrix[i][j+1]
            elif j == 3:
                left += abs(matrix[i][j] - matrix[i][j - 1])
                right += float(matrix[i][j] - matrix[i][(j + 1) % 4])/3
            if (i == 0 or i == 3) and (j!=0 and j!=3):
                left += abs(matrix[i][j] - matrix[i][j-1])
                right += matrix[i][j] - matrix[i][j+1]
            if (j == 0 or j == 3) and (i != 0 and i != 3):
                up += abs(float(matrix[i][j] - matrix[i - 1][j]) / 4)
                down += float(matrix[i][j] - matrix[i + 1][j]) / 4
            if (j!=0 and j!=3) and (i != 0 and i != 3):
                left += abs(matrix[i][j] - matrix[i][j - 1])
                right += matrix[i][j] - matrix[i][j + 1]
                up += abs(float(matrix[i][j] - matrix[i - 1][j]) / 4)
                down += float(matrix[i][j] - matrix[i + 1][j]) / 4
            #print(left + right + up + down)
            res += left + right + up + down
    return res
        
        
def heuristic_value(state):
    b=np.asarray(state)
    a=np.reshape(b,(4,4))
    hval=0
    for i in range(len(a)):
        for j in range(len(a)):
            pv=(4*i)+j+1
            x,y=0,0
            if a[i][j]!=pv:
                x=abs(int(a[i][j]/4)-i)
                if j!=int(a[i][j]%4)-1:
                    y=int(a[i][j]%4) 
            if x==1 or x==3:
                x=1
            if y==1 or y==3:
                y=1
            hval+=x+y
            #print(hval,a[i][j])
    return int(hval/4)


# Priority queue initialization
def initializePriorityQueue(initial_board):
    q = PriorityQueue()
    q.put((0, (initial_board, "")))
    return q

# The solver! - using BFS right now
def solve(initial_board):
    fringe = initializePriorityQueue(initial_board)
    while fringe.qsize() > 0:
        (state, route_so_far) = fringe.get()[1]
        #print(print_board(tuple(state)))
        #print("Val", heuristic_value(state) + len((route_so_far).split()))
        #print(state)
        #print(heuristic_value(state) + len((route_so_far).split()))
        for (succ, move) in successors(state):
            if is_goal(succ):
                return (route_so_far + " " + move)
            #print(print_board(tuple(succ)))
            #print("Val", heuristic_value(succ) + len((route_so_far + " " + move).split()))
            fringe.put((heuristic_value(succ) + len((route_so_far + " " + move).split()), (succ, route_so_far + " " + move)))
    return False

# test cases
start_state = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        start_state += [int(i) for i in line.split()]

if len(start_state) != 16:
    print "Error: couldn't parse start state file"

route = solve(tuple(start_state))

print route
