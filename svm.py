# from data_preprocessiong import *

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import CCA


import numpy as np
a = np.array([1,2,3,4,5])
b = np.array([1,2,4,4,6])

# c = [x for x in (a!=b)]
# print(c)

d = (a>0) * 1
print(d)