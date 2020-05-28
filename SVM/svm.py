from data_preprocessiong import *

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

import numpy as np


# hyper parameters
N_ESTIMATORS = 8
TEST_PROPORTION = 0.2


def Cal_Accuracy(clf, X, y):
    """Calculate the accuracy of the clf on dataset X, y.

    Arguments:
    ---
    clf : classifier model
        trained classifier
    X : src_matrix
        input data
    y : ndarray
        label

    Returns:
    ---
        accu : float
    """    
    pred = clf.predict(X)
    corr = np.sum([int(x) for x in (pred==y)])
    accu = corr / len(y)
    return accu


def Cal_F1(clf, X, y):
    """Calculate the F1 value of the clf on dataset X, y.

    Arguments:
    ---
    clf : classifier model
        trained classifier
    X : src_matrix
        input data
    y : ndarray
        label

    Returns:
    ---
        F1 : float
    """    
    pred = clf.predict(X)
    # accuracy
    corr = np.sum([int(x) for x in (pred==y)])
    accu = corr / len(y)
    
    # recall
    positive_samples = 0
    TP = 0
    for i in range(len(y)):
        if y[i] > 0:
            positive_samples += 1
            if pred[i] > 0:
                TP += 1
    reca = TP / positive_samples

    F1 = 2*accu*reca / (accu+reca)
    return F1


def SVM():
    """Use the sklearn libary SVM model to predict. Convert the multi class problem to
    binary class problem.
    """  
    data = pd.read_csv(dataFile_path)
    _train_X, _train_y, _test_X, _test_y = Split_Data_Set(data, TEST_PROPORTION)

    train_X, test_X = Vectorize_X(_train_X, _test_X, 'TFIDF')

    for k in range(NUM_CLASSES):
        train_y, test_y = Vectorize_y(_train_y, _test_y, k)
        # define SVM model
        model = svm.SVC(decision_function_shape='ovo')
        # train the model
        clf = model.fit(train_X, train_y)

        print(f'category_{k}, {CATEGORIES_[k]} >> train score:', clf.score(train_X, train_y))
        print(f'category_{k}, {CATEGORIES_[k]} >> test score:', clf.score(test_X, test_y))



def Multiclass_SVM():
    """Use sklearn libary OneVsRestClassifier model, to solve the problem as an entirety, without
    resorting to binarize the problem.
    """ 
    data = pd.read_csv(dataFile_path)
    _train_X, _train_y, _test_X, _test_y = Split_Data_Set(data, TEST_PROPORTION)

    # vectorization
    train_X, test_X = Vectorize_X(_train_X, _test_X, 'TFIDF')
    train_y, test_y = Multiclass_Label(train_y, test_y)

    # define SVM model, parameter 'decision_function_shape' decide if it is a multiclass problem
    model = svm.SVC(decision_function_shape='ovr')
    # train the model
    clf = model.fit(train_X, train_y)

    print('train score:', clf.score(train_X, train_y))
    print('test score:', clf.score(test_X, test_y))


if __name__ == "__main__":
    pass