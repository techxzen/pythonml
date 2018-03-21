# @file  : test_naiveBayes.py
# @author: chencheng
# @date  : 20180315

# email classification
# ham and spam(junk mail)
# process:
# 1. fi.read() get all the words as DataX, and DataY
# 2. create a wordList
# 3. sperate trainingSet and testSet
# 4. for trainingSet, calc P(Y), P(X|Y)
# 5. words2vec
# 6. for testSet, calc P(Y|X) using bayes, get error rate


import os

import re

import random

import naiveBayes


def getFileList(dirName):
    for root, dirs, files in os.walk(dirName):
        break
    for i in range(len(files)):
        files[i] = root + files[i]
    return files


def getHamData(dataSetX, dataSetY, dirName):
    fileList = getFileList(dirName)
    for fileName in fileList:
        fi = open(fileName, 'r+')
        content = fi.read()
        fi.close()
        dataSetX.append(content)
        dataSetY.append(0)


def getSpamData(dataSetX, dataSetY, dirName):
    fileList = getFileList(dirName)
    for fileName in fileList:
        fi = open(fileName, 'r+')
        content = fi.read()
        fi.close()
        dataSetX.append(content)
        dataSetY.append(1)


def getDataSet():
    rawSetX = []
    rawSetY = []
    hamDir = '../../data/naiveBayes_data/email/ham/'
    spamDir = '../../data/naiveBayes_data/email/spam/'
    getHamData(rawSetX, rawSetY, hamDir)
    getSpamData(rawSetX, rawSetY, spamDir)
    # regular expression
    regEx = re.compile('\\W*')
    dataSetX = []
    for item in rawSetX:
        dataSetX.append(regEx.split(item))
    return dataSetX, rawSetY


def createVocabList(dataSetX):
    vocabSet = set([])
    # set 2 list
    for item in dataSetX:
        vocabSet = vocabSet | set(item)
    return list(vocabSet)


def words2Vec(words, vocabList):
    # zero value list
    vec = [0] * len(vocabList)
    for word in words:
        if(word in vocabList):
            vec[ vocabList.index(word) ] = 1
        else:
            print('no such word %s in vocabulary list!' %(word))
    return vec


def main():
    # data 
    dataSetX, dataSetY = getDataSet()
    # get word list
    vocabList = createVocabList(dataSetX)    
    
    # create trainSet, testSet
    testIndices = []
    totalNum = len(dataSetX)
    trainIndices = range(totalNum)
    for _ in range(int(totalNum * 0.2)):
        randNumber = int(random.uniform(0, len(trainIndices)) )
        testIndices.append( trainIndices[randNumber] )
        trainIndices.remove( trainIndices[randNumber] )

    # words2vec
    trainVecX = []
    trainSetY = []
    testVecX  = []
    testSetY  = []
    for idx in trainIndices:
        trainVecX.append(words2Vec(dataSetX[idx], vocabList))
        trainSetY.append(dataSetY[idx])
    for idx in testIndices:
        testVecX.append(words2Vec(dataSetX[idx], vocabList))
        testSetY.append(dataSetY[idx])

    # train
    p0, p1, p0s, p1s = naiveBayes.train_c2(trainVecX, trainSetY)
    print(p0)
    print(p1)
    # test
    errorCount = 0.0
    totalCount = 0.0
    for idx in range(len(testVecX)):
        totalCount += 1
        testVec = testVecX[idx]
        c = naiveBayes.classify_c2(testVec, p0, p1, p0s, p1s)
        if(c != testSetY[idx]):
            errorCount += 1
    print('errorCount / totalCount: %d / %d = %f' %(errorCount, totalCount, errorCount/totalCount))



if __name__ == "__main__":
    main()
