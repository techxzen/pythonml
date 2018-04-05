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
            ''' https://matplotlib.org/api/colors_api.html '''
            ''' 
            (r,g,b), all channels value is 0~1.
            (1,0,0) means R
            (0,1,0) means G
            '''
            color.append((1,0,0)) #each channel is 0~1, not 0~255
        else:
            color.append((0,1,0))
            
    x = np.linspace(x1_min, x1_max, 10)
    
    ''' figure '''
    figure_num = len(weights)
    fig, ax = plt.subplots(figure_num, 1)
    ''' axes '''
    for idx in range(figure_num):
        ax[idx].scatter(x1, x2, c=color)
        y = theline(weights[idx], x)
        ax[idx].plot(x, y, 'b-')

        ax[idx].set(title='test')
        ax[idx].set(xlabel='x1')
        ax[idx].set(ylabel='x2')
    ''' show '''
    plt.show()

def plotRecord(record):
    y0 = record[:,0]
    y1 = record[:,1]
    y2 = record[:,2]
    num = len(record)
    x = range(num)
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(x, y0)
    ax[1].plot(x, y1)
    ax[2].plot(x, y2)
    plt.show()
    
def main():
    dataSetX, dataSetY = loadData('../../data/logisticRegression/testSet.txt')
    ''' batch gd '''
    print('batch gd')
    weights_0, record_0 = logReg.batch_GD(dataSetX, dataSetY, 0.001, 50000, True)
    print(weights_0)
    plotRecord(record_0)
    
    ''' stochastic gd '''
    print('stochastic gd')
    weights_1, record_1 = logReg.stochastic_GD(dataSetX, dataSetY, 0.001, 500, True)
    print(weights_1)
    plotRecord(record_1)
    
    ''' improved stochatic gd, faster, stable '''

    weights = [weights_0, weights_1]
    plot(dataSetX, dataSetY, weights)
    
    
if __name__ == "__main__":
    main()
