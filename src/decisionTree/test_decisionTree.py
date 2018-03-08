# @file  : decisionTree_test.py
# @author: chencheng
# @date  : 20180303

import decisionTree

def readLensesData():
    fi = open('../../data/decisionTree_data/lenses.txt','r+')
    lines = fi.readlines()
    dataSetX = []
    dataSetY = []
    for line in lines:
        datalist = line.strip().split('\t')
        dataY = datalist[-1]
        dataX = datalist[0:-1]
        dataSetX.append(dataX)
        dataSetY.append(dataY)
    print('Lenses data read!')
    return dataSetX, dataSetY


def main():
    dataSetX, dataSetY = readLensesData()
    featureList = ['age', 'prescript', 'astigmatic', 'tearRate']
    threshold = 0
    # create decisionTree
    tree = decisionTree.createTree(dataSetX, dataSetY, featureList, threshold)
   
    # save
    import pickle
    fo = open('tree.file','w+')
    pickle.dump(tree, fo)
    fo.close()

    # load and use 
    inX = ['young', 'myope', 'yes', 'normal']
    fi = open('tree.file')
    tree = pickle.load(fi)
    fi.close()
    inY = decisionTree.classify(tree, featureList, inX)
    print(inX)
    print(inY)

if __name__ == "__main__":
    main()
