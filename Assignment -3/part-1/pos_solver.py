###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
#
#
# For MCMC Gibbs Sampling
# Transition Probabilities: In this case transition probabilities for two precedent parts of speech tags
# must be considered from thrid word in a sentence.
# P(S(i) | S(i-1), S(i-2)) where Si is the Parts of Speech tag for i'th word in a sentence.
# Earlier for HMM we have built transition probabilities dictionary of dictionaries. In this case since we
# have to store another variable(parts of speech tag), we need to add one more layer.
# This will be helpful to obtain transition probability by just accessing corresponding keys in the dictionary.
# For example, P(noun | verb, adj) can be obtained by looking at dict[noun][verb][adj]
# If in case there are no probabilities for a particular sequence then a min probability is assigned
# for computation feasibility.
# Sampling:
# For a sentence we need to generate large number of samples with parts of speech tags for each word.
# Here we are generating 100 samples with 50 samples left out as warm-up.
# Tried for different number of samples ranging from 100 to 2000, but the accuracies did not significantly change.
# So keeping in mind the running time too, 100 samples were considered in the end.
# Initially we are starting with the word predictions using HMM method.Instead of starting with random tags
# this is a better choice and chances are high for the convergence to happen quicker.
# For a sample, sentence is looped over for words
#   For each word
#       Tags for all other words are kept constant.
#       Probability distribution of current word's parts of speech is calculated.
#       This is done by calculating probabilities for current word being each of 12 speech tags.
#       For each of 12 parts of speech tags
#           Probability of current word being current parts of speech tag is calculated.
#       Using the distribution a tag is picked for current word and is updated to be used for next sample.
#   This updated one will be iterated over for generating next sample.
# In this way we generate 1000 samples to let the distributions become stationary, not storing any of them
# Then the samples are stored for next 1000 iterations and these are stored.
# From these stored values we choose the speech tag with maximum number of occurrences for particular word and
# assign the same.

####

import random
import math
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        self.minProb = float(1) / float(10**10)
        self.emission_count = {}
        self.transition_count = {}
        self.transition_count_double = {}
        self.pos_count = {}
        self.emission_prob = {}
        self.transition_prob = {}
        self.transition_prob_double = {}
        self.pos_prob = {}
        self.unique_words = {}
        self.unique_pos = {}
        self.totalWordCount = 0
        self.noOfRecords = 0
        self.posterior_simple = 0
        self.posterior_HMM = 0

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        #print(sentence, label)
        if model == "Simple":
            posterior = 0
            for i in range(len(sentence)):
                if sentence[i] in self.emission_prob and label[i] in self.emission_prob[sentence[i]]:
                    posterior += math.log(self.emission_prob[sentence[i]][label[i]])
                posterior += math.log(self.pos_prob[label[i]])
            return posterior
        elif model == "Complex":
            # print(self.get_probability(sentence, label))
            posterior_complex = self.get_probability(sentence, label)
            return posterior_complex
        elif model == "HMM":
            posterior = 0
            for i in range(len(sentence)):
                if sentence[i] in self.emission_prob and label[i] in self.emission_prob[sentence[i]]:
                    posterior += math.log(self.emission_prob[sentence[i]][label[i]])
                if i == 0:
                    posterior += math.log(self.transition_prob[label[i]][" "])
                else:
                    posterior += math.log(self.transition_prob[label[i]][label[i-1]])
            return posterior
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        for w,s in data:
            self.totalWordCount += len(w)
            self.noOfRecords += 1
            for i in range(len(w)):
                if w[i] not in self.unique_words:
                    self.unique_words[w[i]] = True
                if s[i] not in self.unique_pos:
                    self.unique_pos[s[i]] = True
                if w[i] in self.emission_count:
                    if s[i] in self.emission_count[w[i]]:
                        self.emission_count[w[i]][s[i]] += 1
                    else:
                        self.emission_count[w[i]][s[i]] = 1
                else:
                    self.emission_count[w[i]] = {}
                    self.emission_count[w[i]][s[i]] = 1
                if s[i] in self.pos_count:
                    self.pos_count[s[i]] += 1
                else:
                    self.pos_count[s[i]] = 1
                if i == 0:
                    if s[i] in self.transition_count:
                        if " " in self.transition_count[s[i]]:
                            self.transition_count[s[i]][" "] += 1
                        else:
                            self.transition_count[s[i]][" "] = 1
                    else:
                        self.transition_count[s[i]] = {}
                        self.transition_count[s[i]][" "] = 1

                else:
                    if s[i] in self.transition_count:
                        if s[i-1] in self.transition_count[s[i]]:
                            self.transition_count[s[i]][s[i-1]] += 1
                        else:
                            self.transition_count[s[i]][s[i-1]] = 1
                    else:
                        self.transition_count[s[i]] = {}
                        self.transition_count[s[i]][s[i-1]] = 1
                if i>=2:
                    if s[i] in self.transition_count_double:
                        if s[i-1] in self.transition_count_double[s[i]]:
                            if s[i-2] in self.transition_count_double[s[i]][s[i-1]]:
                                self.transition_count_double[s[i]][s[i-1]][s[i-2]] += 1
                            else:
                                self.transition_count_double[s[i]][s[i-1]][s[i-2]] = 1
                        else:
                            self.transition_count_double[s[i]][s[i-1]] = {}
                            self.transition_count_double[s[i]][s[i-1]][s[i-2]] = 1
                    else:
                        self.transition_count_double[s[i]] = {}
                        self.transition_count_double[s[i]][s[i-1]] = {}
                        self.transition_count_double[s[i]][s[i-1]][s[i-2]] = 1
        # Counts to Probablities

        # ---------------- Emission Probabilities -------------------------
        for word in self.unique_words.iterkeys():
            for pos in self.unique_pos.iterkeys():
                if word not in self.emission_prob:
                    self.emission_prob[word] = {}
                if pos in self.emission_count[word]:
                    self.emission_prob[word][pos] = float(self.emission_count[word][pos]) / float(self.pos_count[pos])
                else:
                    self.emission_prob[word][pos] = self.minProb

        # ----------------- Transition Probabilities ------------------------
        for pos in self.unique_pos.iterkeys():
            if pos not in self.transition_prob:
                self.transition_prob[pos] = {}
            for prev_pos in self.unique_pos.iterkeys():
                if prev_pos in self.transition_count[pos]:
                    self.transition_prob[pos][prev_pos] = float(self.transition_count[pos][prev_pos]) / float(self.totalWordCount - self.noOfRecords)
                else:
                    self.transition_prob[pos][prev_pos] = self.minProb

        # transition probabilities for mcmc case where in each pos depends on the pos of two precedent words.

        for pos in self.unique_pos.iterkeys():
            if pos not in self.transition_prob_double:
                self.transition_prob_double[pos] = {}
            for prev1_pos in self.unique_pos.iterkeys():
                if prev1_pos in self.transition_count_double[pos]:
                    if prev1_pos not in self.transition_prob_double[pos]:
                        self.transition_prob_double[pos][prev1_pos] = {}
                    for prev2_pos in self.unique_pos.iterkeys():
                        if prev2_pos in self.transition_count_double[pos][prev1_pos]:
                            self.transition_prob_double[pos][prev1_pos][prev2_pos] = float(self.transition_count_double[pos][prev1_pos][prev2_pos]) / float(self.totalWordCount - (2*self.noOfRecords))
                        else:
                            self.transition_prob_double[pos][prev1_pos][prev2_pos] = self.minProb
                else:
                    if prev1_pos not in self.transition_prob_double[pos]:
                        self.transition_prob_double[pos][prev1_pos] = {}
                    for prev2_pos in self.unique_pos.iterkeys():
                        self.transition_prob_double[pos][prev1_pos][prev2_pos] = self.minProb

        for pos in self.unique_pos.iterkeys():
            if " " in self.transition_count[pos]:
                self.transition_prob[pos][" "] = float(self.transition_count[pos][" "]) / float(self.noOfRecords)
            else:
                self.transition_prob[pos][" "] = self.minProb

        # ----------------- POS Probabilities ------------------------
        for pos in self.unique_pos.iterkeys():
            self.pos_prob[pos] = float(self.pos_count[pos]) / float(self.totalWordCount)
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        pos_predicted = []
        for word in sentence:
            max_prob = -1
            max_pos = "."
            for pos in self.unique_pos.iterkeys():
                if word not in self.emission_prob:
                    x = self.minProb
                else:
                    x = self.emission_prob[word][pos]
                if self.pos_prob[pos]*x > max_prob:
                    max_prob = self.pos_prob[pos]*x
                    max_pos = pos
            pos_predicted.append(max_pos)

        return pos_predicted

    def complex_mcmc(self, sentence):
        # pos_predicted = ["noun" for i in range(len(sentence))]
        pos_predicted = self.hmm_viterbi(sentence)
        init_predicted = pos_predicted
        k = 0
        samples = []
        for k in range(100):
            sample_predicted = self.gibbs_sample(sentence, init_predicted)
            if(k>50):
                samples.append(sample_predicted)
            k += 1
        sample_cols = list(zip(*samples))
        pos_predicted = [max(set(a), key=a.count) for a in sample_cols]
        return pos_predicted

    def gibbs_sample(self, sentence, init_predicted):
        pos_sample = list(init_predicted)
        for i in range(len(sentence)):
            pos = []
            pos_prob = []
            for curr_pos in self.unique_pos.iterkeys():
                pos.append(curr_pos)
                pos_sample[i] = curr_pos
                curr_pos_prob = self.get_probability(sentence, pos_sample)
                pos_prob.append(math.exp(curr_pos_prob))
            norm_sum = sum(pos_prob)
            if(norm_sum == 0):
                pos_sample[i] = pos[0]
            else:
                temp_prob = [x/norm_sum for x in pos_prob]
                pos_sample[i] = np.random.choice(pos, p=temp_prob)
        return pos_sample

    def get_probability(self, sentence, pos_predicted):
        prob = 0
        for l in range(len(sentence)):
            word = sentence[l]
            pos = pos_predicted[l]
            if word not in self.emission_prob:
                x = self.minProb
            else:
                x = self.emission_prob[word][pos]
            prob += math.log(x)
            if l==0:
                prob += math.log(self.pos_prob[pos])
            elif l==1:
                pos_prev1 = pos_predicted[l-1]
                prob += math.log(self.transition_prob[pos][pos_prev1])
            else:
                pos_prev1 = pos_predicted[l-1]
                pos_prev2 = pos_predicted[l-2]
                prob += math.log(self.transition_prob_double[pos][pos_prev1][pos_prev2])
            # print(prob)
        return prob


    def hmm_viterbi(self, sentence):
        pos_predicted = ["noun" for i in range(len(sentence))]
        matrix = [[(0, ".", ".") for i in range(12)] for j in range(len(sentence))]
        pos_index = {}

        j = 0
        for pos in self.unique_pos.iterkeys():
            pos_index[pos] = j
            j += 1


        j = 0
        for pos in self.unique_pos.iterkeys():
            if sentence[0] not in self.emission_prob:
                x = self.minProb
            else:
                x = self.emission_prob[sentence[0]][pos]
            matrix[0][j] = (self.transition_prob[pos][" "] * x, pos, pos)
            j += 1

        for i in range(1, len(sentence)):
            k = 0
            for pos in self.unique_pos.iterkeys():
                j = 0
                max_prob = -1
                prev_pos = "."
                for pos1 in self.unique_pos.iterkeys():
                    if matrix[i - 1][j][0] * self.transition_prob[pos][matrix[i - 1][j][1]] > max_prob:
                        max_prob = matrix[i - 1][j][0] * self.transition_prob[pos][matrix[i - 1][j][1]]
                        prev_pos = matrix[i - 1][j][1]
                    j += 1
                if sentence[i] not in self.emission_prob:
                    x = self.minProb
                else:
                    x = self.emission_prob[sentence[i]][pos]
                matrix[i][k] = (max_prob * x, pos, prev_pos)
                k += 1

        # --------------------- Backtracking for the solution ------------------------------------
        j = 0
        max_prob = -1
        final_pos = "."
        for pos in self.unique_pos.iterkeys():
            if matrix[len(sentence) - 1][j][0] > max_prob:
                max_prob = matrix[len(sentence) - 1][j][0]
                final_pos = matrix[len(sentence) - 1][j][1]
                #pos_predicted[len(sentence) - 1] = final_pos
            j += 1
        if max_prob !=0:
            self.posterior_HMM = math.log(max_prob)
        l = len(sentence) - 1
        for i in range(len(sentence)):
            pos_predicted[l] = final_pos
            final_pos = matrix[l][pos_index[final_pos]][2]
            l = l - 1
        return pos_predicted


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

