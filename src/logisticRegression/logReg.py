# @author: chencheng
# @title : logReg.py
# @data  : 20180326

import numpy as np

''' activation function '''
def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result


''' batch gradient descent '''
def batch_GD(dataSetX, dataSetY, alpha=0.001, iteration=500, debug=False):
    example_num = dataSetX.shape[0]
    feature_num = dataSetX.shape[1]
    
    weights = np.ones((feature_num,1))
    dataSetY = dataSetY.reshape((example_num, 1))
    
    record = []
    ''' iterations '''
    for i in range(iteration):
        tmpY = sigmoid( np.dot(dataSetX, weights) )
        delta = tmpY - dataSetY
        weights = weights - alpha * np.dot(dataSetX.T, delta)
        if(debug):
            record.append(weights)
    return weights, np.array(record)


''' stochastic gradient descent '''
def stochastic_GD(dataSetX, dataSetY, alpha=0.01, iteration=200, debug=False):
    example_num = dataSetX.shape[0]
    feature_num = dataSetX.shape[1]
    
    weights = np.ones((feature_num,1))
    dataSetY = dataSetY.reshape((example_num, 1))
    
    record = []
    for _ in range(iteration):
        for j in range(example_num):
            tmpY = sigmoid( np.dot(dataSetX[j], weights) )
            delta = tmpY - dataSetY[j]
            weights = weights - alpha * delta * dataSetX[j].reshape(feature_num, 1)
            if(debug):
                record.append(weights)
    return weights, np.array(record)
    
    
''' mini-batch gradient descent '''
def mini_batch_GD():
    return 0