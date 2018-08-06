'''
*******************************************************************************
*       Filename:  adaboost.py
*    Description:  py file
*       
*        Version:  1.0
*        Created:  2018-08-05
*         Author:  chencheng
*
*        History:  initial draft
*******************************************************************************
'''

import numpy as np


def loadSimpData():
    data = [[1.0,  2.1],
			[2.0,  1.1],
			[1.3,  1.0],
			[1.0,  1.0],
			[2.0,  1.0]]
    label = [1, 1, -1, -1, 1]
	
    dataSetX = np.array(data)
    dataSetY = np.array(label)
    dataSetY = np.reshape(dataSetY, (5,1))
    return dataSetX, dataSetY


def treeClassify(dataArray, dimen, threshold, inEq):
    ret = np.ones((dataArray.shape[0], 1))
    ''' when is negative sample '''
    if (inEq == "lt"):
        index = dataArray[:,dimen] < threshold
        ret[index] = -1;
    else:
        index = dataArray[:,dimen] >= threshold
        ret[index] = -1

    return np.array(ret)


def buildTree(dataSetX, dataSetY, dWeight):
    ''' The tree is a single-layer-decision-tree,
        find the most valuable feature, and threshold '''
    m, n = dataSetX.shape

    minError = np.inf
    bestTree = {}

    for i in range(n):
        maxFeatureValue = np.max(dataSetX[:, i])
        minFeatureValue = np.min(dataSetX[:, i])

        numSteps = 10
        stepSize = (maxFeatureValue - minFeatureValue) / numSteps

        for j in range(-1, np.int(numSteps + 2)):
            threshold = minFeatureValue + np.float(j) * stepSize

            for inequal in ['lt', 'gt']:
                errorArray = np.zeros((m, 1))
                out = treeClassify(dataSetX, i, threshold, inequal)
                print(out)
                #print(dataSetY)
                errorArray[out != dataSetY] = 1

                weightedError = np.dot(dWeight.T, errorArray)
                #print(dWeight.T)
                #print(errorArray)
                #print(weightedError)
                print("split: dim %d, threshold %.2f, inequal: %s -> weightedError: %f" \
                             %(i, threshold, inequal, weightedError))
                if (weightedError < minError):
                    minError = weightedError
                    bestEst  = out.copy()
                    bestTree["dim"] = i
                    bestTree["thresh"] = threshold
                    bestTree["ineq"] = inequal
    return bestTree, minError, bestEst


def main():
    dataX, dataY = loadSimpData()
    print(dataX)
    print(dataY)

    D = np.ones((5,1)) / 5
    print(D)

    bestTree, minError, bestEst = buildTree(dataX, dataY, D)

    print((bestTree, minError, bestEst))


if __name__ == "__main__":
	main()