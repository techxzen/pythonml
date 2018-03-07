# @file  : decisionTree.py
# @author: chencheng
# @date  : 20180303

# ID3 / C4.5 algorithm
# it is a recursive algorithm
# input : D(training set), A(feature set), threshold
# output: T(tree)
# process:
# 1. if all samples in D belong to the same class C,
#      T is a single-node-tree, the node flag is C.
#      return T
# 2. if A is null, 
#      T is a single-node-tree, the node flag is the majority.
#      return T
# 3. for a feature Ai in A, compute the g(D, Ai) = H(D) - H(D|Ai)
#      find the max one Ag
# 4. if g(D, Ag) < threshold, 
#      T is a single-node-tree, the node flag is the majority.
#      return T
# 5. else:
#      Ag as the Root node
#      split the D to Di, according to the Ag value
#      for every Di:
#           Ti = createTree(Di)
#           add Ti to the Root
#      return T

import numpy as np
import math


def calcEntropy(dataSetY):
    num = len(dataSetY)
    # stastic data
    count = {}
    for dataY in dataSetY:
        count[dataY] = count.get(dataY, 0) + 1
    # compute
    entropy = 0
    for item in count.items():
        prob = float(item[1]) / num 
        entropy += - prob * math.log(prob)
    return entropy


def splitDataSet(dataSetX, dataSetY, featureIdx, featureValue):
    subDataSetX = []
    subDataSetY = []
    for i in range(len(dataSetY)):
        if(dataSetX[i][featureIdx] == featureValue):
            dataX = dataSetX[i].tolist()
            tmpX = dataX[:featureIdx]
            tmpX.extend(dataX[featureIdx+1:])
            subDataSetX.append(tmpX)
            subDataSetY.append(dataSetY[i])
    return subDataSetX, subDataSetY


def maxInformationGain(dataSetX, dataSetY):
    # H(D)
    entropyD = calcEntropy(dataSetY)
    # find the max gain
    maxIdx = 0
    maxGain = 0
    for i in range( dataSetX.shape[1] ):
        # H(D|Ai)
        features = [item[i] for item in dataSetX]
        uniqValue = set(features)
        newEntropy = 0
        for value in uniqValue:
            subDataSetX, subDataSetY = splitDataSet(dataSetX, dataSetY, i, value)
            prob = float(len(subDataSetY)) / float((len(dataSetY)))
            newEntropy += prob * calcEntropy(subDataSetY)
        Gain = entropyD - newEntropy
        if(Gain > maxGain):
            maxGain = Gain
            maxIdx  = i
    return maxGain, maxIdx


def createTree(dataSetX, dataSetY, featureList, threshold):
    # convertTo np.array
    dataSetX = np.array(dataSetX)
    dataSetY = np.array(dataSetY)

    # if all belong to same class
    countD = {}
    for dataY in dataSetY:
        countD[dataY] = countD.get(dataY, 0) + 1
    countDList = countD.items()
    sortedCountD = sorted(countDList, \
                    key=lambda item:item[1], reverse=True)
    totalNum = dataSetY.shape[0]
    maxClassNum = sortedCountD[0][1]
    maxClass    = sortedCountD[0][0]
    if totalNum == maxClassNum:
        return maxClass
    # if A is null
    if dataSetX.shape[1] == 0:
        return maxClass
    # others, split
    tree = {}
    maxGain, maxIdx = maxInformationGain(dataSetX, dataSetY)
    if(maxGain < threshold):
        return maxClass
    else:
        featureName = featureList[maxIdx]
        
        tree[ featureName ] = {}

        features= [example[maxIdx] for example in dataSetX]

        uniqValue = set(features)
        for value in uniqValue:
            tmpSetX, tmpSetY \
                  = splitDataSet(dataSetX, \
                                 dataSetY, \
                                  maxIdx, value)
            tmpFeatureList = featureList[:maxIdx]
            tmpFeatureList.extend(featureList[maxIdx:])
            tree[featureName][value] = createTree(tmpSetX, tmpSetY, \
                                                  tmpFeatureList,\
                                                  threshold)
        return tree
        

def classify(tree, featureList, inX):
    print((0,tree))
    if( not isinstance(tree, dict) ):
        return tree
    else:
        print(tree)
        # determin the idx
        feature = tree.keys()[0]
        idx = 0
        for idx in range(len(featureList)):
            if(featureList[idx] == feature):
                break
        value = inX[idx]
        return classify(tree[feature][value], featureList, inX)
