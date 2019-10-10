#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2018)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print(im.size)
    #print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]

    return exemplars

def emission(test_letter, letter):
    emission_prob = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(test_letter)):
        for j in range(len(test_letter[i])):
            if test_letter[i][j] == train_letters[letter][i][j] and test_letter[i][j] == '*':
                emission_prob += math.log(0.95)
                tp += 1
            elif test_letter[i][j] == train_letters[letter][i][j] and test_letter[i][j] == ' ':
                emission_prob += math.log(0.65)
                fn += 1
            elif test_letter[i][j] != train_letters[letter][i][j] and test_letter[i][j] == ' ':
                emission_prob += math.log(0.35)
                tn += 1
            else:
                emission_prob += math.log(0.05)
                fp += 1
    '''if tp == 0:
        tp = 1e-100
    #print("tp, fp, tn, fn", tp, fp, tn, fn)
    precision = float(tp) / float(tp + fp)
    #print("precision", precision)
    recall = float(tp) / float(tp + fn)
    #print("recall", recall)
    emission_prob = float(precision * recall) / float(precision + recall)
    emission_prob *= 2'''
    return emission_prob


# main program
(train_img_fname, train_txt_fname, test_img_fname) = (sys.argv[1], sys.argv[2], sys.argv[3])
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

transition_count = {}
transition_prob = {}
data = read_data(train_txt_fname)
scaler = 1.3

#--------------------- Taking transition counts ---------------------------
for words,pos in data:
    for word in words:
        '''if word == "''" or word == ".":
            continue'''
        prev_letter = " "
        for letter in word:
            if letter in transition_count:
                if prev_letter in transition_count[letter]:
                    transition_count[letter][prev_letter] += 1
                else:
                    transition_count[letter][prev_letter] = 1
                transition_prob[letter][prev_letter] = 1
            else:
                transition_count[letter] = {}
                transition_prob[letter] = {}
                transition_count[letter][prev_letter] = 1
                transition_prob[letter][prev_letter] = 1
            
            prev_letter = letter
        
        if " " in transition_count:
            if prev_letter in transition_count[" "]:
                transition_count[" "][prev_letter] += 1
            else:
                transition_count[" "][prev_letter] = 1
            transition_prob[" "][prev_letter] = 1
        else:
            transition_count[" "] = {}
            transition_prob[" "] = {}
            transition_count[" "][prev_letter] = 1
            transition_prob[" "][prev_letter] = 1

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

for letter in transition_count:
    total_count = sum(transition_count[letter].values())
    for succ in transition_count[letter]:
        transition_prob[letter][succ] = math.log(float(transition_count[letter][succ]) / float(total_count)) * scaler
        #print(succ, letter, transition_prob[succ][letter])

'''for letter1 in letters:
    for letter2 in letters:
        if letter2 in transition_prob:
            if letter1 in transition_prob[letter2]:
                transition_prob[letter2][letter1] = transition_count[letter2][letter1] / 5112
            else:
                transition_prob[letter2] = {}
                transition_prob[letter2][letter1] = 1e-100
        else:
            transition_prob = {}
            transition_prob[letter2] = {}
            transition_prob[letter2][letter1] = 1e-100'''
        
#count = sum(transition_count[" "].values())

for letter in letters:
    if letter in transition_count[" "]:
        transition_prob[" "][letter] = math.log(float(transition_count[" "][letter]) / float(sum(transition_count[" "].values()))) * scaler
    else:
        transition_prob[" "][letter] = math.log(1e-10) * scaler
    #transition_prob[" "][letter] = math.log(float(1) / float(72)) * scaler

'''i = 0
for letter in letters:
    if letter not in transition_prob:
        transition_prob[letter] = {}
        transition_prob[letter][" "] = math.log(1e-10) * scaler
    else:
        if i < 52:
            transition_prob[letter][" "] = math.log(float(3) / float(176))
        else:
            transition_prob[letter][" "] = math.log(float(1) / float(176))
        transition_prob[letter][" "] = math.log(float(1) / float(72)) * scaler
    i += 1'''

#transition_prob[" "]["."] = math.log(0.2)*scaler
#transition_prob[" "][","] = transition_prob[" "]["."]
transition_prob[" "][" "] = math.log(1e-320) * scaler


#-------------------------------- Simple Model ----------------------------------------------
simple_answer = ''
for i in range(len(test_letters)):
    max_prob = -1000
    max_letter = 'A'
    for letter in letters:
        emissionProb = emission(test_letters[i], letter)
        if emissionProb > max_prob:
            max_prob = emissionProb
            max_letter = letter
    simple_answer += max_letter

print("Simple: ", simple_answer)

#-------------------------------- HMM -------------------------------------------------
matrix = [[(0, 'A') for i in range(72)] for j in range(len(test_letters))]

j = 0
for letter in letters:
    if letter in transition_prob:
        #print(transition_prob[letter][" "], emission(test_letters[0], letter), letter)
        matrix[0][j] = (transition_prob[letter][" "] * scaler + emission(test_letters[0], letter), letter)
    elif letter.lower() in transition_prob:
        matrix[0][j] = (transition_prob[letter.lower()][" "] * scaler+ emission(test_letters[0], letter), letter)
    else:
        matrix[0][j] = (math.log(1e-200) * scaler, letter)
    j += 1


for i in range(1, len(test_letters)):
    for j in range(len(letters)):
        max_prob = -1000000
        prev_letter = 'A'
        for k in range(len(letters)):
            if letters[j] in transition_prob and letters[k] in transition_prob[letters[j]]:
                x = transition_prob[letters[j]][letters[k]] * scaler
            elif letters[j].lower() in transition_prob and letters[k] in transition_prob[letters[j].lower()]:
                x = transition_prob[letters[j].lower()][letters[k]] * scaler
            else:
                x = math.log(1e-200) * scaler
            if matrix[i-1][k][0] + x > max_prob:
                max_prob = matrix[i-1][k][0] + x
                prev_letter = matrix[i-1][k][1]
            
        matrix[i][j] = (max_prob + emission(test_letters[i], letters[j]), prev_letter + letters[j])

#------------------------ Backtracking for solution ---------------------------
answer = ''
max_prob = -1000000
final_letter = ''
index = 0
for j in range(len(letters)):
    if matrix[len(test_letters) - 1][j][0] > max_prob:
        max_prob = matrix[len(test_letters) - 1][j][0]
        final_letter = letters[j]
        answer = final_letter
        index = j

print("Viterbi: ", matrix[len(test_letters) - 1][index][1])
print("Final Answer: ")
print(matrix[len(test_letters) - 1][index][1])


