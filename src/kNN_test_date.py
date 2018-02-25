# chencheng
# 20180225
# classify person for dating

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def getData():
    filename = "../data/datingTestSet.txt"
    fi = open(filename, 'r+')
    lines = fi.readlines()
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

def main():
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

if __name__ == "__main__":
    main()
