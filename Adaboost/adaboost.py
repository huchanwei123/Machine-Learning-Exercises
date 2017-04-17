import numpy as np
import math
import csv
import os, sys
import pdb

def load_data(DATA_PATH):
    label = []
    data = []
    with open(DATA_PATH, newline='') as csvfile:
        read_in = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in read_in:
            row_split = str(row)[2:-2].split(",")
            label.append(row_split[0])
            data.append(row_split[1:len(row_split)])

    label = [int(i) for i in label]
    for i in range(len(data)):
        data[i] = [int(j) for j in data[i]]
    label = np.asarray(label)
    data = np.asarray(data)

    # Change the label number to -1 and +1
    unique_label = np.unique(label)
    label[label == unique_label[0]] = -1
    label[label == unique_label[1]] = 1
    return data, label

def stump_classify(data, which_attr, threshold):
    output = np.ones(data.shape[0])
    output[data[:, which_attr] <= threshold] = -1
    return output

def build_stump(data, label, D_weight):
    feature_num, feature_size = data.shape
    Steps = 10.0
    dim_list = []
    threshold_list = []
    bestStump = {}
    bestClassEst = np.zeros((feature_num))
    minError = math.inf
    for i in range(feature_size):
        min_attr = min(data[:, i])
        max_attr = max(data[:, i])
        step_size = (max_attr - min_attr) / Steps
        for j in range(0, int(feature_num)):
            threshold = min_attr + j * step_size
            predict_val = stump_classify(data, i, threshold)
            error_arr = np.ones(feature_num)
            error_arr[predict_val == label] = 0
            weight_error = np.dot(D_weight, error_arr)
            if weight_error < minError:
                minError = weight_error
                bestClassEst = predict_val.copy()
                bestStump['dim'] = i
                bestStump['thresh'] = threshold

    return bestStump, minError, bestClassEst

def Adaboost(data, label, phase, Iteration):
    feature_num, feature_size = data.shape
    D = np.ones((feature_num))/feature_num
    bestStumpArr = []
    aggClassEst = np.zeros(feature_num)
    for i in range(Iteration):
        bestStump, error, bestClassEst = build_stump(data, label, D)
        alpha = float(0.5 * math.log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        bestStumpArr.append(bestStump)
        expon = np.multiply(-1 * alpha * label, bestClassEst)
        z = 2 * math.sqrt(error * (1 - error))
        D = np.multiply(D, np.exp(expon)) / z
        aggClassEst += alpha * bestClassEst
        aggError = np.multiply(np.sign(aggClassEst) != label, np.ones((feature_num)))
        errorRate = sum(aggError)/feature_num
        if(phase == 'training'):
            print('Training iteration ' , i+1, ', training error: ', errorRate)
        if errorRate == 0.00:
            break
    return bestStumpArr

def adaClassify(data, label, phase, classifierArr):
    feature_num, feature_size = data.shape
    aggClassEst = np.zeros(feature_num)
    errorRate = 0
    for i in range(len(classifierArr)):
        classEst = stump_classify(data, classifierArr[i]['dim'], classifierArr[i]['thresh'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        aggError = np.multiply(np.sign(aggClassEst) != label, np.ones((feature_num)))
        errorRate = sum(aggError)/feature_num

    predict_output = np.sign(aggClassEst)
    correct = 0
    for i in range(len(label)):
        if(int(predict_output[i]) == label[i]):
            correct = correct + 1
    accuracy = (float(correct)/len(label)) * 100
    if(phase == 'validation'):
        return errorRate
    elif(phase == 'testing'):
        return accuracy
    else:
        return 0
