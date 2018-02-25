# helper functions common used
# Author: chencheng
# Date  : 20180225

# auto Normalize
# Process:
# 1. find the max, min
# 2. new value = (value - min)/(max - min)

import numpy as np

def autoNormalize(dataSet):
    dataSet = np.array(dataSet)
    dataNum = dataSet.shape[0]
    # 1. find the max, min
    maxs = np.max(dataSet, axis = 0)
    mins = np.min(dataSet, axis = 0)
    # 2. new value = (value - min) / (max - min)
    ranges = maxs - mins
    tmp = dataSet - np.tile(mins, (dataNum, 1))
    normDataSet = tmp / np.tile(ranges, (dataNum, 1))

    return normDataSet, ranges, mins    
