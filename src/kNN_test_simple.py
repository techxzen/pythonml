# test kNN
# chencheng
# 20180225

import numpy as np
import kNN

#group = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
group = [[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]]
labels = np.array(['A', 'A', 'B', 'B'])

inX = [0, 0]
k = 3

label = kNN.classify(inX, group, labels, k)
print(label)
