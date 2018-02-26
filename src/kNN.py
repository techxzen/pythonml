# kNN algorithm
# Author: chencheng
# Date  : 20180225

# The kNN algorithm: 
# input : trainingSetX/Y, inputX, the k Param
# output: outputY
# Process flow:
# 1. Find the nearest k samples in trainingSet for input X
# 2. Find the most class in k samples as the output Y

import numpy as np


def findNearestKSamples(inX, trainingSetX, k):
    # Euler distance
    tmp = np.tile(inX, (trainingSetX.shape[0], 1))
    tmp = (tmp - trainingSetX) ** 2
    tmp = np.sum(tmp, axis = 1)
    distances = tmp ** 0.5
    idxRank = np.argsort(distances)
    return idxRank[0:k]


def classify(inX, trainingSetX, trainingSetY, k):
    # 0. Error check
    # ensure it is np.array type
    inX  = np.array(inX)
    trainingSetX = np.array(trainingSetX)
    trainingSetY = np.array(trainingSetY) 
    if(trainingSetX.shape[0] != trainingSetY.shape[0]):
        print("ERROR: trainingSetX is not equal to trainingSetY!")
        exit(0)
    trainingSetSize = trainingSetY.shape[0]

    # 1. find the nearest k sample in trainingSet
    kSamplesIdx = findNearestKSamples(inX, trainingSetX, k)

    # 2. find the most class in k samples as the output Y
    classCount = {}
    for idx in kSamplesIdx:
        label = trainingSetY[idx]
        classCount[label] = classCount.get(label,0) + 1
    sortedClassCount = sorted(classCount.items(),\
                     key=lambda item:item[1], reverse=True)
    
    return sortedClassCount[0][0]
