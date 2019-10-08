#!/bin/python
# 09/17/2018

# The following is the approach and conditions we considered for this problem
# In a state, group one is considered and if there are less than 3 students in it,
# all the combinations with one student groups are generated.
# The same is repeated with rest of the groups. In this way all the combinations of different
# group sizes are generated.
# Initial state is structured with one student in a separate group. And minimum effort variable
# is stored and updated with each of the successor having effort less than the current minimum effort.
# From all the successors only the successors with effort less than
# minimum effort till then are considered and inserted into fringe.
# So effectively for each state we are restricting the number of successors
# based on the effort calculated for corresponding state. This will be same as DFS algorithm
# in addition to eliminating states that are assumed to not produce minimum goal state eventually.
# This is because the total number of states to be considered for a reasonably larger number of
# students into groups of sizes(1,2,3) are huge and checking all of them is merely not possible.
# We did try considering all the states for 50 students list but the number of combinations were in millions.
# And the same algorithm with BFS did give a better result(to a small extent) but comparing the execution time
# to that of DFS approach we decided not to go with BFS one.

import sys
import copy
import time
start_time = time.time()

# Initiate groups with each student in a separate group
def initGroups(students):
    for i in range(len(students)):
        groups_init.append([students[i]])
    return


# To find the effort for grading assignments for current state
# Number of groups in given state is calculated and its product with parameter k is taken
# Input: state
# Output: grading effort value
def getGradingEffort(state):
    team_count = len(state)
    return team_count*k


# To find the effort for complaints regarding group size
# Group size preference of each student are stored in studet_groupsize_map dictionary
# For each student in each group, if the preference does not match with the group assignment currently
# it is considered effort is incremented by 1
# Effort is not considered in case of '0' given as preference.
def getGroupSizeComplaintEffort(state):
    cost = 0
    for group in state:
        for student in group:
            if student_groupsize_map[student] != 0 and student_groupsize_map[student] != len(group):
                # cost += abs(student_groupsize_map[student] - len(group))
                cost += 1
    return cost


# To find the effort for complaints regarding preferred students not being in the group
# and the also for the ones assigned in the group as non preferred
# student_pref_map and student_not_pref_map dictionary variables contain the respective data for each student
# For each student in a group both these values are checked with the currenlty assignede groups
# and corresponding weights m/n are incremented in the effort.
def getPrefStudentComplaintEffort(state):
    pref_cost = 0
    not_pref_cost = 0
    student_check = []
    for group in state:
        for student in group:
            # if not(student in student_check):
            if student in student_pref_map:
                for pref_student in student_pref_map[student]:
                    if not(pref_student in group):
                        pref_cost += n
            if student in student_not_pref_map:
                for not_pref_student in student_not_pref_map[student]:
                    if not_pref_student in group:
                        not_pref_cost += m
                # student_check.append(student)
    return pref_cost, not_pref_cost


# To find total effort required for given state formation(groups of students)
def effort(state):
    grading_effort = getGradingEffort(state)
    # print('Grading Effort :' , grading_effort)
    groupsize_complaint_effort = getGroupSizeComplaintEffort(state)
    # print('Group size complaint :', groupsize_complaint_effort)
    (pref_student_complaint_effort,not_pref_student_complaint_effort) = getPrefStudentComplaintEffort(state)
    # print('Pref complaint :',pref_student_complaint_effort)
    effort_total = grading_effort + groupsize_complaint_effort + pref_student_complaint_effort + not_pref_student_complaint_effort
    return effort_total

# Function to generate set of successor states for a given state
def successors(state):
    successors = []
    for i in range(0,len(state)):
        if len(state[i]) < 3:
            # group_size = len(state[i])
            for j in range(i+1, len(state)):
                temp_state = copy.deepcopy(state)
                if len(temp_state[j]) == 1:
                    temp_state[i].append(temp_state[j][0])
                    temp_state.remove(temp_state[j])
                    effort_current = effort(temp_state)
                    successors.append([temp_state,effort_current])
    return successors

# Main function that generates groups and returns a
# group that requires minimum effort from implemented algorithm

def solve(initial_board):
    effort_init = effort(initial_board)
    fringe = [[initial_board, effort_init]]
    # i = 0
    # j = 0
    effort_vals = []
    min_effort = effort_init
    min_effort_state = initial_board
    while len(fringe) > 0:
        (state, effort_state) = fringe.pop()
        for (succ, effort_succ) in successors(state):
            # i += 1
            effort_vals.append(effort_succ)
            if effort_succ < min_effort:
                # j += 1
                min_effort = effort_succ
                min_effort_state = succ
                fringe.append([succ, effort_state])
    return (min_effort_state,min_effort)

# Read file name and parameters from the command line
file_path = sys.argv[0]
input = sys.argv[1]
k = int(sys.argv[2])
m = int(sys.argv[3])
n = int(sys.argv[4])

# Initiate variables for list of students and mapped values
student_list = []
student_num_id = {}
student_groupsize_map = {}
student_pref_map = {}
student_not_pref_map = {}
groups_init = []

# Read input text file and store the values to corresponding variables
with open(input, 'r') as file:
    student_id = 1
    for line in file:
        info = line.split()
        student_list.append(info[0])
        student_num_id[info[0]] = student_id
        student_groupsize_map[info[0]] = int(info[1])
        if info[2] != '_':
            student_pref_map[info[0]] = info[2].split(',')
        if info[3] != '_':
            student_not_pref_map[info[0]] = info[3].split(',')
        student_id += 1

# Generate initial state by arranging each student in a group
initGroups(student_list)

(final_state, final_state_effort) = solve(groups_init)

print '\n'.join(' '.join(s for s in p) for p in final_state)
print(final_state_effort)

# print(time.time() - start_time)
