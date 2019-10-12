#!/usr/bin/python2
import sys
import numpy as np
from random import randint
from scipy import stats
import math
import json
from collections import Counter
# from sklearn.metrics import accuracy_score
from shutil import copyfile
import random
import copy
from Queue import PriorityQueue
#from queue import Queue
import cPickle as pickle
import os


class Node():
    index = ''
    left = ''
    right = ''
    threshold = 120
    answer = -1
    level = 1

    def __init__(self):
        self.index = '1'
        self.answer = -1
        self.level = 1
        self.left = None
        self.right = None


# Creates an instance of the class DisneyCharacter

# def read_data(fname):
#     train_dict = {}
#     train = []
#     labels = []
#     file = open(fname, 'r')
#     for line in file:
#         samp = [p for p in line.split()]
#         img = samp[0]
#         label = samp[1]
#         temp = np.array(samp[1:]).astype(int)
#         labels.append(label)
#         train_dict[img] = temp
#         train.append(temp)
#
#     return train_dict, np.array(train), np.array(labels).astype(int)


def giniIndex(index, data, data_shape):
    p_0, p_90, p_180, p_270 = 0, 0, 0, 0
    entropy_left, entropy_right = 0, 0
    splitValue = np.median(data[:, index])
    # print("split val", splitValue)
    leftData = data[np.where(data[:, index] <= splitValue)]
    rightData = data[np.where(data[:, index] < splitValue)]
    leftData_shape = leftData.shape[0]
    rightData_shape = rightData.shape[0]
    if leftData_shape > 0:
        p_0 = float(data[np.where(leftData[:, 0] == 0)].shape[0]) / float(leftData_shape)
        p_90 = float(data[np.where(leftData[:, 0] == 90)].shape[0]) / float(leftData_shape)
        p_180 = float(data[np.where(leftData[:, 0] == 180)].shape[0]) / float(leftData_shape)
        p_270 = float(data[np.where(leftData[:, 0] == 270)].shape[0]) / float(leftData_shape)
        # total_count_left = count_0 + count_90 + count_180 + count_270
        if p_0 == 0:
            p_0 = 1
        if p_90 == 0:
            p_90 = 1
        if p_180 == 0:
            p_180 = 1
        if p_270 == 0:
            p_270 = 1
        entropy_left = (-1 * p_0 * math.log(p_0)) + (-1 * p_90 * math.log(p_90)) + (-1 * p_180 * math.log(p_180)) + (
                    -1 * p_270 * math.log(p_270))
    #  + (-1 * p_90 * math.log(p_90[, 2])) + (-1 * p_180 * math.log(p_180[, 2])) + (-1 * p_270 * math.log(p_270[, 2]))

    if rightData_shape > 0:
        p_0 = float(data[np.where(rightData[:, 0] == 0)].shape[0]) / float(rightData_shape)
        p_90 = float(data[np.where(rightData[:, 0] == 90)].shape[0]) / float(rightData_shape)
        p_180 = float(data[np.where(rightData[:, 0] == 180)].shape[0]) / float(rightData_shape)
        p_270 = float(data[np.where(rightData[:, 0] == 270)].shape[0]) / float(rightData_shape)
        # total_count_right = count_0 + count_90 + count_180 + count_270

        if p_0 == 0:
            p_0 = 1
        if p_90 == 0:
            p_90 = 1
        if p_180 == 0:
            p_180 = 1
        if p_270 == 0:
            p_270 = 1

        entropy_right = (-1 * p_0 * math.log(p_0)) + (-1 * p_90 * math.log(p_90)) + (-1 * p_180 * math.log(p_180)) + (
                    -1 * p_270 * math.log(p_270))

    total_entropy = (float(leftData_shape) / float(data_shape)) * entropy_left + (
                float(rightData_shape) / float(data_shape)) * entropy_right
    return total_entropy


def getBestSplitfeature(features, data, data_shape):
    min = 100
    minFeature = ''
    if data_shape == 0:
        return None
    for feature in features:
        gini = giniIndex(feature.index, data, data_shape)
        # print("gini", gini)
        if gini < min:
            min = gini
            minFeature = feature
    temp = copy.copy(minFeature)
    return temp


def hasUniqueTarget(data):
    targets = data[:, 0]
    uniqueTarget = {}
    uniqueTarget[targets[0]] = 1
    for i in range(1, len(targets)):
        # res = bool(res) ^ bool(targets[i])
        if not targets[i] in uniqueTarget:
            uniqueTarget[targets[i]] = 1
        else:
            uniqueTarget[targets[i]] += 1

    if len(uniqueTarget.items()) == 1:  # res == False:
        return (True, targets[0])

    maxFreq = -1
    maxKey = targets[0]
    for key, val in uniqueTarget.items():
        if val > maxFreq:
            maxFreq = val
            maxKey = key

    return (False, maxKey)

def train_rf(train_fl, model_fl):
    # train_file = sys.argv[1]
    train_dict, train_data, train_labels, train_imgs = read_data(train_fl)
    id1 = 0
    var = 0
    trees = []
    file = os.path.dirname(os.path.abspath(__file__))
    fp = open(os.path.join(file, model_fl), "wb")
    for i in range(100):
        id1 = 0
        var = 0
        rootNode = Node()
        featureIndexes = random.sample(range(1, 192), 14)
        features = []
        for j in range(len(featureIndexes)):
            feature = Node()
            feature.index = featureIndexes[j]
            feature.answer = -1
            features.append(feature)

        rootNode = getBestSplitfeature(features, train_data, 36976)
        rootNode.level = 1
        rootNode.id = var
        rootNode.pid = -1

        queue = PriorityQueue()
        queue.put((rootNode.id, (rootNode, train_data)))

        while queue.qsize() > 0:

            root, data = queue.get()[1]

            if data.shape[0] == 0 or root == None or var > 100000:
                break

            splitValue = np.median(data[:, root.index])
            root.threshold = splitValue

            leftData = data[np.where(data[:, root.index] <= splitValue)]
            rightData = data[np.where(data[:, root.index] > splitValue)]
            leftData_shape = leftData.shape[0]
            rightData_shape = rightData.shape[0]
            hasuniqueTarget, answer = hasUniqueTarget(data)

            if hasuniqueTarget or root.level > 11:

                root.answer = answer
                root.left = None
                root.right = None

            else:
                var += 1
                if leftData_shape > 1:
                    root.left = getBestSplitfeature(features, leftData, leftData_shape)
                    root.left.level = root.level + 1
                    root.left.dir = 'left'
                    id1 += 1
                    root.left.id = id1
                    root.left.pid = root.id

                    queue.put((root.left.id, (root.left, leftData)))

                if rightData_shape > 1:
                    root.right = getBestSplitfeature(features, rightData, rightData_shape)

                    root.right.level = root.level + 1
                    root.right.dir = 'right'
                    id1 += 1
                    root.right.id = id1
                    root.right.pid = root.id

                    queue.put((root.right.id, (root.right, rightData)))

        pickle.dump(rootNode, fp)
        trees.append(rootNode)

    fp.close()


def test_rf(test_fl, model_fl):
    # test_file = "test-data.txt"
    test_dict, test_data, test_labels, test_imgs = read_data(test_fl)
    stored_trees = []
    with (open(model_fl, "rb")) as openfile:
        while True:
            try:
                stored_trees.append(pickle.load(openfile))
            except EOFError:
                break

    predictions = 0
    i = 0
    j = 0
    test_out = []
    for test in test_data:
        votes = {}
        votes[0] = 0
        votes[90] = 0
        votes[180] = 0
        votes[270] = 0

        for currentNode in stored_trees:

            while True:
                if currentNode.right != None and currentNode.left != None:
                    if test[currentNode.index] <= int(currentNode.threshold):
                        if currentNode.left != None:
                            currentNode = currentNode.left
                        else:
                            break
                    else:
                        if currentNode.right != None:
                            currentNode = currentNode.right
                        else:
                            break
                else:
                    if currentNode.answer == -1:
                        i += 1
                    else:
                        votes[currentNode.answer] += 1
                    break
        max_count = -1
        max_label = 0
        if votes[0] > max_count:
            max_count = votes[0]
            max_label = 0
        if votes[90] > max_count:
            max_count = votes[90]
            max_label = 90
        if votes[180] > max_count:
            max_count = votes[180]
            max_label = 180
        if votes[270] > max_count:
            max_count = votes[270]
            max_label = 270

        if max_label == test[0]:
            predictions += 1
        test_out_str = test_imgs[j] + " " + str(max_label)
        test_out.append(test_out_str)
        j += 1
    with open('output_forest.txt', 'w') as file:
        for s in test_out:
            file.write("%s\n" % s)
    print("Overall Accuracy", float(predictions) / float(test_data.shape[0]))


def read_data(fname):
    train_dict = {}
    train =[]
    labels = []
    imgs = []
    file = open(fname, 'r')
    for line in file:
        samp = [p for p in line.split()]
        img = samp[0]
        label = samp[1]
        temp = np.array(samp[1:]).astype(int)
        labels.append(label)
        train_dict[img] = temp
        train.append(temp)
        imgs.append(img)
    return train_dict, np.array(train), np.array(labels).astype(int), imgs



# def adaboost(data, labels):
#     k = 10 # Number of weak classifiers to be considered
#     # n = labels.size
#
#     for i in range(0,k):
#         pass
#


def hypothesis(data, labels, model_file):
    j = 0
    H = {}
    h_wts = []
    n = labels.size
    # Initialize weights vector for each sample
    w = n*[1/float(n)]
    while(j<400):
        err = 0
        p1 = randint(1, 192)
        p2 = randint(1, 192)
        if(not([p1,p2] in H.values())):
            x_p1 = data[data[:, p1] >= data[:, p2]]
            if(x_p1.size > 0):
                p1_pred = stats.mode(x_p1[:, 0])[0][0]
            # print(p1_pred)
            x_p2 = data[data[:, p1] < data[:, p2]]
            if(x_p2.size > 0):
                p2_pred = stats.mode(x_p2[:, 0])[0][0]
            # print(p2_pred)
            for i in range(0,n):
                # print(data[i,0])
                if(data[i, p1] >= data[i, p2]):
                    if(data[i,0] != p1_pred):
                        err += w[i]
                else:
                    if(data[i,0] != p2_pred):
                        err += w[i]
            # print("Error :" , err)
            err = err/sum(w)
            if(err <= 0.75):
                alpha = math.log((1-err)/err) + math.log(3) # alpha = log((1-err)/err)) + log(K -1) , where K is number of classes
                for k in range(0,n):
                    if (data[k, 0] != p1_pred or data[k, 0] != p2_pred):
                        w[k] = w[k] * math.exp(alpha)

                # Normalize weights
                w_sum = sum(w)
                w = [a/w_sum for a in w]
                H[j] = []
                H[j].append([p1, p2])
                H[j].append([p1_pred, p2_pred])
                h_wts.append(alpha)
                j += 1
    # print("Hypotheses :", H)
    # print("Weights :", h_wts)
    model_out(H, h_wts, model_file)

def test_adap(data, labels, images, model_fl):
    hyp, wts = read_model(model_fl, 'adaboost')
    i = 0
    acc = 0
    test_out = []
    for img in data:
        output_class = 4 * [0]
        for key in hyp:
            p1 = hyp[key][0][0] - 1
            p2 = hyp[key][0][1] - 1
            pred_p1 = hyp[key][1][0]
            pred_p2 = hyp[key][1][1]
            if(img[p1] >= img[p2]):
                out = pred_p1/90
            else:
                out = pred_p2/90
            # print(out)
            output_class[int(out)] += wts[int(key)]
        out_pred = output_class.index(max(output_class))
        out_pred = out_pred * 90
        if(out_pred == labels[i]):
            acc += 1
        test_out_str = images[i] + " " + str(out_pred)
        test_out.append(test_out_str)
        i += 1
    with open('output_adaboost.txt', 'w') as file:
        for s in test_out:
            file.write("%s\n" % s)
    print(acc)
    print('Accuracy :', acc/float(labels.size))
    return acc

def model_out(h, w, file):
    hyp_dict = {'adaboost': {'hypotheses' :h, 'weights' :w}}
    with open(file, 'w') as file:
        file.write(json.dumps(hyp_dict))

def read_model(file_name, model):
    with open(file_name, "r") as file:
        data = json.load(file)
        # print("Reading file :" ,data[model])
        hyp = data[model]['hypotheses']
        wts = data[model]['weights']
    return hyp, wts

def nearest_train(train_fl, model_fl):
    copyfile(train_fl, model_fl)

def nearest(x_train, x_test, y_train, y_test, test_imgs, model):
    k=48
    predictions = predict(x_train,x_test,y_train,k, test_imgs, model)
    # for i in range(int(len(x_test))):
    #     predts = predict(x_train,x_test,y_train,k)
    #     predictions = predictions + list(predts)
    # acc = accuracy_score(y_test, predictions)
    # print('Accuracy :', acc)
    # output = np.column_stack((test_imgs, predictions))

def predict(x_train,x_test,y_train, k, imgs, model):
    dists = compute_distances(x_train,x_test)
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    test_out = []
    for i in range(num_test):
        k_closest_y = []
        labels = y_train[np.argsort(dists[i,:])].flatten()
        # find k nearest lables
        k_closest_y = labels[:k]
        c = Counter(k_closest_y)
        y_pred[i] = c.most_common(1)[0][0]
        test_out_str = imgs[i] + " " + str(int(y_pred[i]))
        test_out.append(test_out_str)
    if(model == "nearest"):
        out_file = 'output_nearest.txt'
    else:
        out_file = 'output_best.txt'
    with open(out_file, 'w') as file:
        for s in test_out:
            file.write("%s\n" % s)
    return(y_pred)

def compute_distances(x_train,X):
    num_test=len(X)
    num_train=len(x_train)
    dists=np.zeros((num_test,num_train))
    for i in range(num_test):
        dists[i,:]=np.sum((x_train-X[i,:])**2,axis=1)
    return(dists)

file = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]
if(sys.argv[1] == 'train'):
    # train_file = sys.argv[1]
    # train_file = 'train-data.txt'
    train_dict, train_data, train_labels, train_images = read_data(file)
    if(model == "adaboost"):
        # print(train_data.shape)
        hypothesis(train_data, train_labels, model_file)
    if (model == "nearest" or model == "best"):
        nearest_train(file, model_file)
    if (model == "forest"):
        train_rf(file, model_file)

elif(sys.argv[1] == 'test'):
    test_dict, test_data, test_labels, test_images = read_data(file)
    if(model == "adaboost"):
        test_data = test_data[:, 1:]
        test_adap(test_data, test_labels, test_images, model_file)
    if (model == "nearest" or model == "best"):
        train_dict, train_data, train_labels, train_images = read_data(model_file)
        train_data = train_data[:, 1:]
        test_data = test_data[:, 1:]  # Removing labels for test data
        # test_dict, test_data, test_labels, test_images = read_data(file)
        # test_data = test_data[:, 1:]  # Removing labels for test data
        # print(train_data.shape)
        nearest(train_data, test_data, train_labels, test_labels, test_images, model)
        # print(test_data)
    if(model == "forest"):
        test_rf(file, model_file)
