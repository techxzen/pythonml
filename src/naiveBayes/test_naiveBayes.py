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
    vec = [0]


def main():
    # data 
    dataSetX, dataSetY = getDataSet()
    # get word list
    vocabList = createVocabList(dataSetX)    
    
    # create trainSet, testSet
    testIndices = []
    totalNum = len(dataSetX)
    trainIndices = range(total_num)
    for _ in range(int(totalNum * 0.2)):
        randNumber = int(random.uniform(0, len(trainIndices))
        testIndices.append( trainIndices[randNumber] )
        del( trainIndices[randNumber] )

    # words2vec
    trainVecX = []
    for idx in trainIndices:
        trainVecX.append(words2Vec(dataSetX[idx], vocabList))
    


if __name__ == "__main__":
    main()
