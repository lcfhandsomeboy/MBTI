from data_preprocessiong import *

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import f1_score, accuracy_score

import numpy as np


# hyper parameters
TEST_PROPORTION = 0.2


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
        model = svm.SVC(decision_function_shape='ovr', kernel='poly')
        # train the model
        clf = model.fit(train_X, train_y)

        train_pred = clf.predict(train_X)
        test_pred = clf.predict(test_X)

        print(accuracy_score(train_y, train_pred))
        print(f1_score(train_y, train_pred, pos_label=1, average='binary'))



def Multiclass_SVM():
    """Use sklearn libary OneVsRestClassifier model, to solve the problem as an entirety, without
    resorting to binarize the problem.
    """ 
    data = pd.read_csv(dataFile_path)
    _train_X, _train_y, _test_X, _test_y = Split_Data_Set(data, TEST_PROPORTION)

    # vectorization
    train_X, test_X = Vectorize_X(_train_X, _test_X, 'TFIDF')
    train_y, test_y = Multiclass_Label(_train_y, _test_y)

    # define SVM model, parameter 'decision_function_shape' decide if it is a multiclass problem
    model = svm.SVC(decision_function_shape='ovr', kernel='poly')
    # model = OneVsRestClassifier(svm.SVC(kernel='linear'))

    # train the model
    clf = model.fit(train_X, train_y)

    train_pred = clf.predict(train_X)
    test_pred = clf.predict(test_X)

    print(accuracy_score(train_y, train_pred))
    print(f1_score(train_y, train_pred, average='macro'))


if __name__ == "__main__":
    # SVM()
    Multiclass_SVM()