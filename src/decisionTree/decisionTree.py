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

def maxInformationGain(dataSetX, dataSetY, countDList):
    # H(D)
    num = len(dataSetY)
    prob = countDList / float(num)
    tmp  = - prob * math.log(prob)
    entropy = np.sum(tmp)
    # find the max gain
    for i in range(dataSetX.shape[1]):
        # H(D|Ai)
        for 

def createTree(dateSetX, dataSetY, featureName):
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
    featureIdx, maxGain = maxInformationGain()


def plotTree():
    print('tree ploted')
