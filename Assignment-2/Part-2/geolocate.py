#!/usr/bin/env python
import string
from collections import defaultdict
import sys
import re
import timeit
import math
#from nltk.stem.lancaster import LancasterStemmer

loc_word_count = {}
loc_tweet_count = {}
wordsCount_loc = {}
word_in_train = {}
allWords = {}
idf = {}
tweetCountofWord = {}
# Stop words referenced from https://gist.github.com/sebleier/554280 --> NLTK's list of stop words
stop_words = ["all","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
stop_word = {}
for word in stop_words:
    stop_word[word] = 1

n = sum(1 for line in open(sys.argv[1]))
with open(sys.argv[1], 'r') as file:
    for line in file:
        loc, tweet = line.split(' ', 1)
        loc_tweet_count[loc] = 0
        wordsCount_loc[loc] = 0
        if loc not in loc_word_count:
            loc_word_count[loc] = {}
        for word in re.split(r"[\W_]", tweet):
            word = word.lower()
            tweetCountofWord[word] = 0
            try:
                stop_word[word]
            except KeyError:
                stop_word[word] = 0
            if word != '' and word != ' ' and not word.isdigit() and len(word) > 1 and not stop_word[word]:
                loc_word_count[loc][word] = 0
                try:
                    allWords[word] += 1
                except:
                    allWords[word] = 1

tweet_word = {}
with open(sys.argv[1], 'r') as file:
    for line in file:
        loc, tweet = line.split(' ', 1)
        loc_tweet_count[loc] += 1
        flag = 0
        tweet_word[tweet] = {}
        for word in re.split(r"[\W_]", tweet):
            word = word.lower()
            try:
                tweet_word[tweet][word]
            except:
                tweetCountofWord[word] += 1

            tweet_word[tweet][word] = 1
            if word != '' and word != ' ' and not word.isdigit() and len(word) > 1 and not stop_word[word]:
                loc_word_count[loc][word] += 1
                wordsCount_loc[loc] += 1

for word in allWords.iterkeys():
    if word != '' and word != ' ' and not word.isdigit() and len(word) > 1 and not stop_word[word]:
        idf[word] = {}
        avgFreq = 0
        for loc in loc_tweet_count.iterkeys():
            try:
                avgFreq += loc_word_count[loc][word]
            except KeyError:
                avgFreq += 0
        avgFreq = float(avgFreq)/12
        for loc in loc_tweet_count.iterkeys():
            try:
                idf[word][loc] = loc_word_count[loc][word] / avgFreq
            except KeyError:
                idf[word][loc] = 0#0.01 / avgFreq



tfidf = {}
for loc in loc_tweet_count.iterkeys():
    tfidf[loc] = 0
    for word in loc_word_count[loc].iterkeys():
        tfidf[loc] += loc_word_count[loc][word] * idf[word][loc]
max = 0
freqLoc = ""
for loc in loc_tweet_count.items():
    if loc[1] > max:
        max = loc[1]
        freqLoc = loc[0]

accuracy_count = 0
ntest = sum(1 for line in open(sys.argv[2]))
prob_loc_word = {}

with open(sys.argv[2], 'r') as file:
    for loc in loc_tweet_count.iterkeys():
        prob_loc_word[loc] = {}
    for line in file:
        max_prob = -1
        predicted_loc = ""
        actual_loc, tweet = line.split(' ', 1)
        for loc in loc_tweet_count.iterkeys():
            prob_loc = float(loc_tweet_count[loc]) / float(n)
            prob_word_given_loc = 1

            for word in re.split(r"[\W_]", tweet):
                word = word.lower()
                try:
                    stop_word[word]
                except KeyError:
                    stop_word[word] = 0
                if word != '' and word != ' ' and not word.isdigit() and len(word)>1 and not stop_word[word]:
                    prob_loc_word[loc][word] = 1

                    try:
                        x = loc_word_count[loc][word]
                        prob_loc_word[loc][word] *= ((x * idf[word][loc]) + 1)
                        prob_loc_word[loc][word] = float(prob_loc_word[loc][word]) / float(tfidf[loc] + len(allWords) + 1)

                    except KeyError:
                        x = 0.01
                        prob_loc_word[loc][word] *= ((x) + 1)
                        prob_loc_word[loc][word] = float(prob_loc_word[loc][word]) / float(wordsCount_loc[loc] + len(allWords) + 1)

                    prob_word_given_loc *= prob_loc_word[loc][word]
            if prob_word_given_loc * prob_loc > max_prob:
                max_prob = prob_word_given_loc * prob_loc
                predicted_loc = loc

        if predicted_loc == "":
            predicted_loc = freqLoc
        if predicted_loc == actual_loc:
            accuracy_count += 1
        f = open(sys.argv[3], "a+")
        f.write(predicted_loc + " " + actual_loc + " " + tweet)

    f.close()
    print("Accuracy ---> " + str(accuracy_count) + "/500")
    accuracy_score = float(accuracy_count)/float(ntest) * 100
    print("Accuracy score ---> " + str(accuracy_score))

print("Top words of the location wise: ")
for loc in prob_loc_word.iterkeys():
    print(str(loc) + " ---> " + str(sorted(prob_loc_word[loc], key=prob_loc_word[loc].get, reverse=True)[0:5]))
    print("")
