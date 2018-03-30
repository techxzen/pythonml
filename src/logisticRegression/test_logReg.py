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

def plot():
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    
    ''' figure '''
    fig = plt.figure(0)
    ''' axes '''
    ax = fig.add_subplot(211)
    ax.plot(x, y)
    
    ax = fig.add_subplot(212)
    ax.plot(x*2, y)
    
    ''' show '''
    plt.title('test')
    plt.xlabel('x1')
    plt.ylabel('x2')
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
    plot()
    
    
if __name__ == "__main__":
    main()