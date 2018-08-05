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
	label = [1, 1, -1, -1, -1]
	
	dataSetX = np.array(data)
	dataSetY = np.array(label)
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

    return ret


def buildTree(dataSetX, dataSetY, dWeight):
    ''' The tree is a single-layer-decision-tree,
        find the most valuable feature, and threshold '''
    m, n = dataSetX.shape
    for i in range(n):
        maxFeatureValue = np.max(dataSetX[:, i])
        minFeatureValue = np.min(dataSetX[:, i])

        numSteps = 10
        stepSize = (maxFeatureValue - minFeatureValue) / numSteps

        for j in range(-1, np.int(numSteps + 1)):
            threshold = minFeatureValue + np.float(j) * stepSize

            for inequal in ['lt', 'gt']:
                out = treeClassify(dataSetX, i, threshold, inequal)
                err = np.


def main():
	dataX, dataY = loadSimpData()
	print(dataX)
	print(dataY)


if __name__ == "__main__":
	main()