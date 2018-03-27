# @author: chencheng
# @title : logReg.py
# @data  : 20180326

import numpy as np

''' activation function '''
def sigmoid(x):
	result = 1 / (1 + np.exp(-x))
	return result


''' batch gradient descent '''
def batch_GD(dataSetX, dataSetY, alpha=0.001, iteration=500):
	example_num = dataSetX.shape[0]
	feature_num = dataSetX.shape[1]
	weights = np.ones(feature_num)
	
	''' iterations '''
	for i in range(iteration):
		tmpY = sigmoid(dataSetX * weights.T)
		delta = tmpY.T - dataSetY
		weights = weights - alpha * (delta.T * dataSetX)
	return weights


''' stochastic gradient descent '''
def stochastic_GD():
	return 0


''' mini-batch gradient descent '''
def mini_batch_GD():
	return 0