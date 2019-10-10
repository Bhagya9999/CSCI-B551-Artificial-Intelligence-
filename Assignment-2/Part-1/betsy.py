#!/usr/bin/env python
# Assignment part 1
# Instructor: David Crandall

import sys
import numpy as np
import copy


# #   Minimax alpha beta pruning with iterative deepening is implemented for this problem.


# Evaluation function considers positions of x and o to calculate an evaluation value given a state.
# A set of cases were considered and weights were assigned correspondingly to calculate the value.

def heuristic(state):
    evalue = 0
    c1_max = c1_min = c2_max = c2_min = 0
    # Part of the table to decide goal state. Top n rows of given board state
    upper_board = state[0:n]
    # Lower part of the table. n - n+3 rows
    lower_board = state[n:n + 3]
    max_cnt = (state == max_player).sum()  # sum(l.count('x') for l in state)
    min_cnt = (state == min_player).sum()  # sum(l.count('o') for l in state)
    max_cnt_bottom = (lower_board == max_player).sum()
    min_cnt_bottom = (lower_board == min_player).sum()
    if (is_goal(state)):
        win = is_goal(state)
        if (win == max_player):
            return 99999
        else:
            return -99999
    else:
        # Evaluate difference of pebbles
        evalue = max_cnt - min_cnt
        # Evaluate difference of pebbles in lower part of board.
        evalue = 5 * (max_cnt_bottom - min_cnt_bottom)

        col_diff_value = 0
        row_diff_value = 0
        diag_diff_value = 0

        # Evaluate columns weight
        for i in range(0, n):
            col_diff = sum(upper_board[:, i] == max_player) - sum(upper_board[:, i] == min_player)
            if ('.' in upper_board[:, i] and np.unique(upper_board[:, i]).size == 2):
                col_diff_wt = 15
            else:
                col_diff_wt = 5
            col_diff_value = col_diff_value + col_diff_wt * col_diff
        evalue = evalue + col_diff_value

        # Evaluate rows weight
        for row in upper_board:
            row_diff = sum(row == max_player) - sum(row == min_player)
            if ('.' in row and np.unique(row).size == 2):
                row_diff_wt = 15
            else:
                row_diff_wt = 5
            row_diff_value = row_diff_value + row_diff_wt * row_diff
        evalue = evalue + row_diff_value
        # Evaluate diagonal weights
        diag_diff_value = diag_diff_value + 10 * (sum(np.diag(upper_board) == max_player) - sum(np.diag(upper_board) == min_player))
        diag_diff_value = diag_diff_value + 10 * (sum((np.diag(np.fliplr(upper_board))) == max_player) - sum(
            (np.diag(np.fliplr(upper_board))) == min_player))
        evalue = evalue + diag_diff_value
        return evalue


#   The method generates the possible successors for a given state and current player(pebble x or o)
def successors(state, curr_pebble):
    successors = []
    num_curr_pebbles = (state == curr_pebble).sum()
    for i in range(n):
        succ = copy.copy(state)
        if ('.' in succ[:, i]):
            # Find last index of '.' in the current column
            # Check if max number of pebbles are used and validate drop move
            if(num_curr_pebbles < max_pebbles):
                empty_index = np.where(succ[:, i] == '.')[0][-1]
                succ[:, i][empty_index] = curr_pebble
                successors.append((succ, i + 1))
            state_init = copy.copy(state)
            if (len(set(state_init[:, i])) > 2):
                rotate_pebble = state_init[:, i][-1]
                state_init[:, i] = np.insert(state_init[:, i], 0, '.')[0:n + 3]
                empty_index = np.where(state_init[:, i] == '.')[0][-1]
                state_init[:, i][empty_index] = rotate_pebble
                successors.append((state_init, -(i + 1)))
        else:
            if (len(set(succ[:, i])) > 1):
                succ[:, i] = np.insert(succ[:, i], 0, succ[:, i][-1])[0:n + 3]  # succ[:,i][-1] + succ[:,i][1:n+3]
                successors.append((succ, -(i + 1)))
    return successors


#   Method to represent current state as row
def row_notation(state):
    state_rows = []
    for j in range(n + 3):
        row = [state[i] for i in range(n * j, n * j + n)]
        state_rows.append(row)
    return state_rows


#   Method to display the board state
def display_board(seq):
    rows = (seq[i:n + i] for i in range(0, len(seq), n))
    for r in rows:
        print(r)


#   Method to check if the state has reached goal state.
#   Either win for max or min player.
def is_goal(state):
    # Taking only the top n rows
    # upper_board = map(list, zip(*[state[i][3:n+3] for i in range(0,n)]))
    upper_board = state[0:n]
    # print(upper_board)
    for l in upper_board:
        if ('.' not in l):
            if (np.unique(l).size == 1):
                return l[0]
    for i in range(0, n):
        if ('.' not in upper_board[:, i]):
            if (np.unique(upper_board[:, i]).size == 1):
                return upper_board[:, i][0]
    if (np.unique(np.diag(upper_board)).size == 1 or np.unique(np.diag(np.fliplr(upper_board))).size == 1):
        return np.diag(upper_board)[0]
    return False


#   Method to check if a state has reached goal state or level of depth constraint
#   Returns True if terminal state else False
def is_terminal(state, depth):
    if (depth == depth_const):
        return True
    elif (is_goal(state)):
        return True
    return False

#   Method to perform minimum operation in minimax algorithm operation for a given state, alpha, beta and depth
def minimum(state, depth, alpha, beta):
    if (is_terminal(state, depth)):
        return heuristic(state)
    for s in successors(state, min_player):
        beta = min(beta, maximum(s[0], depth + 1, alpha, beta))
        if (beta <= alpha):
            return beta
    return beta

#   Method to perform maximum operation in minimax algorithm operation for a given state, alpha, beta and depth
def maximum(state, depth, alpha, beta):
    if (is_terminal(state, depth)):
        return heuristic(state)
    for s in successors(state, max_player):
        # print(minimum(s, depth+1, alpha, beta))
        alpha = max(alpha, minimum(s[0], depth + 1, alpha, beta))
        if (alpha >= beta):
            return alpha
    return alpha

#   Main code block to initialize the minimax algorithm using the input board
#   Alpha is taken as -infinity(least possible) and beta as infinity(highest possible)
#   And for itertive deepening depth is increased in steps of 2 between 2 and 10.
#   Resulting best move for each of the depth considered is printed at end of each loop.
def solve(state):
    global depth_const
    set = successors(state, max_player)
    min = -infinity
    best_move = []
    for i in range(2, 8, 2):
        depth_const = i
        for s in set:
            poss = minimum(s[0], 0, -infinity, infinity)
            if (poss > min):
                best_move = s
                min = poss
        print(str(best_move[1]) + ' ' + ''.join(''.join(a for a in row) for row in best_move[0]))
    return

n = int(sys.argv[1])
max_player = sys.argv[2]
seq = sys.argv[3]
# n = 3
# max_player = 'x'
# seq = 'o.oxooxoxxooxxxoxo'
min_player = 'x' if (max_player == 'o') else 'o'
depth_const = 2
infinity = float("inf")
# display_board(seq)
arr_state = np.array(row_notation(seq))
max_pebbles = n * (n + 3) / 2
solve(arr_state)
