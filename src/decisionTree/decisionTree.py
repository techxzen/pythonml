# @file  : decisionTree.py
# @author: chencheng
# @date  : 20180303

# ID3 / C4.5 algorithm
# it is a recursive algorithm
# input : D(training set), A(feature set), threshold
# output: T(tree)
# termination condition:
# 1. if all samples in D belong to the same class C,
#      T is a single-node-tree, the node flag is C.
# 2. if A is null, 
#      T is a single-node-tree, the node flag is the majority.
# process flow:
# 1. for a feature Ai in A, compute the g(D, Ai) = H(D) - H(D|Ai)
#      find the max one Ag
# 2. if Ag < threshold, 
#      T is a single-node-tree, the node flag is the majority.
# 

def createTree(dateSetX, dataSetY):
    tree = {}
    
    print('tree created!')


def plotTree():
    print('tree ploted')
