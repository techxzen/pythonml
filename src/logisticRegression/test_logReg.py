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
		dataSetX.append([1.0, data[0], data[1]])
		dataSetY.append(data[-1])
	dataSetX = np.float32( np.array(dataSetX) )
	dataSetY = np.int32(np.array(dataSetY) )
	return dataSetX, dataSetY

def main():
	dataSetX, dataSetY = loadData('../../data/logisticRegression/testSet.txt')
	# print(dataSetX)	
	weights = logReg.batch_GD(dataSetX, dataSetY)
	
	print(weights)
	
if __name__ == "__main__":
	main()