# chencheng
# 20180225
# classify person for dating

import os

import numpy as np

import matplotlib.pyplot as plt

import helper
import kNN

def getDatingData():
    filename = "../data/datingTestSet.txt"
    fi = open(filename, 'r+')
    lines = fi.readlines()
    fi.close()

    dataSetX = []
    dataSetY = []
    for line in lines:
        alist = line.strip().split()
        if(alist[3] == 'largeDoses'):
            alist[3] = 3
        elif(alist[3] == 'smallDoses'):
            alist[3] = 2
        else:
            alist[3] = 1
        dataSetX.append([float(alist[0]), \
                         float(alist[1]), \
                         float(alist[2]) ])
        dataSetY.append(alist[3])
    return np.array(dataSetX), np.array(dataSetY)

# testSetPercent: 0.1
# k: 3
def datingClassTest(testSetPercent, k):
    dataSetX, dataSetY = getDatingData()
    normDataSetX, ranges, mins = helper.autoNormalize(dataSetX)
    dataSize = normDataSetX.shape[0]
    testSize = int(dataSize * testSetPercent)
    errorCount = 0
    for i in range(testSize):
        inX = normDataSetX[i]
        outY = kNN.classify(inX, normDataSetX[testSize:,:], \
                            dataSetY[testSize:], \
                            k)
        if(outY != dataSetY[i]):
            errorCount = errorCount + 1
    errorRate = errorCount / float(testSize)
    return errorRate

def profiling():
    perSet = [0.1, 0.2, 0.3]
    kSet   = np.arange(1,20,2)
    resultSet = []
    for per in perSet:
        for k in kSet:
            rate = datingClassTest(per, k)
            resultSet.append((per, k, rate))
    sortedResultSet = sorted(resultSet, key = lambda item:item[2])
    print(sortedResultSet[0])


def plotDatingClass():
    dataSetX, dataSetY = getData()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSetX[:,1], dataSetX[:,2], \
                    s = 15.0*np.array(dataSetY), \
                    c = 15.0*np.array(dataSetY))
    plt.title("test")
    plt.xlabel("time used to play games/ percent", fontsize=14)
    plt.ylabel("ice cream used per week/ volume", fontsize=14)
    plt.show()

# ------------------ digits recognition ------------- #
def getVector(filename):
    fi = open(filename, 'r+')
    lines = fi.readlines()
    fi.close()
    vec = []
    for i in range(32):
        line = lines[i]
        for j in range(32):
            vec.append( int(line[j]) )
    return vec

def getTrainingSet(dirname):
    for root, dirs, files in os.walk(dirname):
        break
    trainingSetX = []
    trainingSetY = []
    for filename in files:
        label = filename.strip('.txt').split('_')[0]
        label = int(label)
        filename = root + filename
        inX = getVector(filename)
        trainingSetX.append(inX)
        trainingSetY.append(label)
    return trainingSetX, trainingSetY

def getDigitsData(k):
    dirname = '../data/digits/trainingDigits/'
    trainingSetX, trainingSetY = getTrainingSet(dirname)

    errorCount = 0
    dirname = '../data/digits/testDigits/'
    for root, dirs, files in os.walk(dirname):
        break
    for filename in files:
        header = filename.strip('.txt')
        label = header.split('_')[0]
        label = int(label)
        filename = root +filename
        inX = getVector(filename)
        
        result = kNN.classify(inX, trainingSetX, trainingSetY, k)        
        print((label, result)) 
        if(label != result):
            errorCount += 1
    totalCount = len(files)
    errorRate = errorCount/float(totalCount)
    print((errorRate, errorCount, totalCount))
    return errorRate

def main():
    # plotDatingClass()
    # profiling()
    print( getDigitsData(3) )

if __name__ == "__main__":
    main()
