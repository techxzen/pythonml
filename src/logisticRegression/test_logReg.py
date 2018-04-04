# @author: chencheng
# @title : test_logReg.py
# @data  : 20180326

import numpy as np
import logReg

import matplotlib.pyplot as plt

def loadData(path):
    dataSetX = []
    dataSetY = []
    ''' read from file '''
    fi = open(path, 'r+')
    lines = fi.readlines()
    fi.close()
    for line in lines:
        data = line.strip().split('\t')
        dataSetX.append([1.0, float(data[0]), float(data[1])])
        dataSetY.append(int(data[2]))
    return np.array(dataSetX), np.array(dataSetY)

def theline(weights, x1):
    x2 = (weights[0] + weights[1] * x1) / (-weights[2])
    return x2

def plot(dataSetX, dataSetY, weights):
    x1 = dataSetX[:,1]
    x2 = dataSetX[:,2]
    x1_max = np.max(x1)
    x1_min = np.min(x1)
    
    color = []
    for item in dataSetY:
        if(item == 0):
            color.append((1,0,0)) #each channel is 0~1, not 0~255
        else:
            color.append((0,1,0))
            
    x = np.linspace(x1_min, x1_max, 10)
    y = theline(weights, x)
    
    ''' figure '''
    fig, ax = plt.subplots(1,1)
    ''' axes '''
    ax.plot(x, y, 'b-')
    ax.scatter(x1, x2, c=color)

    ax.set(title='test')
    ax.set(xlabel='x1')
    ax.set(ylabel='x2')
    ''' show '''
    plt.show()


def main():
    dataSetX, dataSetY = loadData('../../data/logisticRegression/testSet.txt')
    ''' batch gd '''
    print('batch gd')
    weights = logReg.batch_GD(dataSetX, dataSetY, 0.001, 500)
    print(weights)
    ''' stochastic gd '''
    print('stochastic gd')
    weights = logReg.stochastic_GD(dataSetX, dataSetY, 0.001, 500)
    print(weights)
    ''' improved stochatic gd, faster, stable '''
    plot(dataSetX, dataSetY, weights)
    
    
if __name__ == "__main__":
    main()