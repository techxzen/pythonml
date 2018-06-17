'''
recommendation system, using svd
singular value decompostion
'''

import numpy as np


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]


def loadExData2():
    data = [[2,0,0,4,4,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,5],
            [0,0,0,0,0,0,0,1,0,4,0],
            [3,3,4,0,3,0,0,2,2,0,0],
            [5,5,5,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,5,0,0,5,0],
            [4,0,4,0,0,0,0,0,0,0,5],
            [0,0,0,0,0,4,0,0,0,0,4],
            [0,0,0,0,0,0,5,0,0,5,0],
            [0,0,0,3,0,0,0,0,4,5,0],
            [1,1,2,1,1,2,1,0,4,5,0]]
    return data


def cosSim(inA, inB):
    num   = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num/denom)


def standEst(dataMat, user, simMeas, item):
    n           = np.shape(dataMat)[1]
    simTotal    = 0.0
    ratSimTotal = 0.0

    ''' 
    for this item, find the if any scored item that simulate him
    '''
    for j in range(n):
        userRating = dataMat[user, j]
        if (userRating == 0):
            continue
        overlap = np.nonzero(np.logical_and(dataMat[:,item].A > 0,
                                         dataMat[:,j].A > 0))[0]
        if (len(overlap) == 0):
            similarity = 0
        else:
            similarity = simMeas(dataMat[overlap, item], dataMat[overlap, j])
        simTotal += similarity
        ratSimTotal += similarity * userRating

    if (simTotal==0):
        return 0
    else:
        return ratSimTotal/simTotal


def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    # print(Sig4)
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    print(xformedItems.shape)
    for j in range(n):
        userRating = dataMat[user, j]
        if (userRating == 0 or j == item):
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        # print('the %d and %d similarity is : %f' %(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
        if (simTotal == 0):
            return 0
        else:
            return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]

    if (len(unratedItems) == 0):# to give any product a score in list
        return 'you rated everything'

    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))

    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]



def main():
    ''' load data '''
    Data = loadExData()

    ''' svd '''
    U, Sigma, V = np.linalg.svd(Data)
    # print(U)
    # print(Sigma)
    # print(V)

    ''' reconstruct, 
        original 5, reduce to 3
    '''
    Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    x = U[:,:3] * Sig3 * V[:3,:]
    # print(np.int32(np.round(x)))

    ''' recommendation system '''    
    myMat = np.mat(Data)
    myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
    myMat[3,3] = 2
    # print(myMat)
    # print(recommend(myMat, 2))

    ''' recommendation system using SVD '''
    Data = loadExData2()
    myMat = np.mat(Data)
    U, Sigma, V = np.linalg.svd(myMat)
    print(Sigma)
    print(recommend(myMat, 1, estMethod=svdEst))

if __name__ == "__main__":
    main()
