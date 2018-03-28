# @author: chencheng
# @title : test_logReg.py
# @data  : 20180326

import numpy as np
import logReg


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

def main():
	dataSetX, dataSetY = loadData('../../data/logisticRegression/testSet.txt')
	weights = logReg.batch_GD(dataSetX, dataSetY, 0.001, 500)
	print(weights)
	
if __name__ == "__main__":
	main()