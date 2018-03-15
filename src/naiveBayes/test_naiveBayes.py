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
# 5. for testSet, calc P(Y|X) using bayes, get error rate


import os


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
    dataSetX = []
    dataSetY = []
    hamDir = '../../data/naiveBayes_data/email/ham/'
    spamDir = '../../data/naiveBayes_data/email/spam/'
    getHamData(dataSetX, dataSetY, hamDir)
    getSpamData(dataSetX, dataSetY, spamDir)
    print(len(dataSetX))
    print(len(dataSetY))


def main():
    # ham dir
    getDataSet()
    


if __name__ == "__main__":
    main()
