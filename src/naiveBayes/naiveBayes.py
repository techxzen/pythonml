# @file  : naiveBayes.py
# @author: chencheng
# @date  : 20180311

# it is commonly used in text classification
# 1. given a dataset(x=text(context), y=class)
# 2. create a vocabulary, usually the 

import numpy as np


def train_c2(trainVecX, trainSetY):
    trainVecX = np.array(trainVecX)
    vocabNum  = trainVecX.shape[1]
    trainNum  = trainVecX.shape[0]
    # p(y)
    p_y0 = 0.0
    p_y1 = 0.0
    # p(xi=1|y=0), p(xi=1|y=1)
    p0s = np.zeros(vocabNum)
    p1s = np.zeros(vocabNum)
    
    for idx in range(trainNum):
        if(trainSetY[idx] == 0):
            p_y0 += 1
            p0s += trainVecX[idx]
        else:
            p_y1 += 1
            p1s += trainVecX[idx]

    p0s   = p0 / p_y0
    p1s   = p1 / p_y1
    p_y1 = p_y1 / trainNum

    return p_y0, p_y1, p0s, p1s


def classify_C2(vec, p_y0, p_y1, p0, p1):
    #p(y=0|x) = p(x|y=0) * p(y=0) / p(x)
    prob0 = p_y0
    for idx in len(vec):
        if(vec[idx] == 1):
            prob0 *= p0[idx]
        else:
            prob0 *= (1 - p0[idx])

    #p(y=1|x) = p(x|y=1) * p(y=1) / p(x)
    prob1 = p_y1
    for idx in len(vec):
        if(vec[idx] == 1):
            prob1 *= p1[idx]
        else:
            prob1 *= (1 - p1[idx])

    # compare
    if(prob0 > prob1):
        return 0
    else
        return 1
