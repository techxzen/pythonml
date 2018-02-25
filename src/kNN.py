# kNN algorithm
# Author: chencheng
# Date  : 20180225

# The kNN algorithm: 
# input : dataSetX/Y, inputX, the k Param
# output: outputY
# Process flow:
# 1. Find the nearest k samples in dataSet for input X
# 2. Find the most class in k samples as the output Y

import numpy as np


def findNearestKSamples(inX, dataSetX, k):
    # Euler distance
    tmp = np.tile(inX, (dataSetX.shape[0], 1))
    tmp = (tmp - dataSetX) ** 2
    tmp = np.sum(tmp, axis = 1)
    distances = tmp ** 0.5
    idxRank = np.argsort(distances)
    return idxRank[0:k]


def classify(inX, dataSetX, dataSetY, k):
    # 0. Error check
    # ensure it is np.array type
    inX  = np.array(inX)
    dataSetX = np.array(dataSetX)
    dataSetY = np.array(dataSetY) 
    if(dataSetX.shape[0] != dataSetY.shape[0]):
        print("ERROR: dataSetX is not equal to dataSetY!")
        exit(0)
    dataSetSize = dataSetY.shape[0]

    # 1. find the nearest k sample in dataSet
    kSamplesIdx = findNearestKSamples(inX, dataSetX, k)

    # 2. find the most class in k samples as the output Y
    classCount = {}
    for idx in kSamplesIdx:
        label = dataSetY[idx]
        classCount[label] = classCount.get(label,0) + 1
    sortedClassCount = sorted(classCount.items(),\
                     key=lambda item:item[0], reverse=True)
    
    return sortedClassCount[0][0]
