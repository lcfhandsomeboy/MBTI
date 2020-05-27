from data_preprocessiong import *

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier


# hyper parameters
N_ESTIMATORS = 10
TEST_PROPORTION = 0.3




def My_AdaBoost():
    """Use the AdaBoost model written by myself to predict. Convert the multi class problem to
    binary class problem.
    """    
    class My_AdaBoost_Classifier():
        """Implement a simple AdaBoost model on my own.

            Argument:
            ---
            weak_clf : sklearn classifier
                the weak classifier of AdaBoost, usually use decision tree classifier.
            
            n_estimators : int
                the number of weak classifier

            train_X : csr_matrix

            test_X : csr_matrix

            train_y : csr_matrix

            test_y : csr_matrix
        """    
        def __init__(self, weak_clf, n_estimators, train_X, train_y, test_X, test_y):     
            self.weak_clf = weak_clf
            self.n_estimators = n_estimators
            self.train_X = train_X
            self.train_y = train_y
            self.test_X = test_X
            self.test_y = test_y

            #initialize weights
            self.w = np.ones(len(self.train_y)) / len(self.train_y)

            # initialize prediction
            self.pred_train, self.pred_test = [np.zeros(len(self.train_y)), np.zeros(len(self.test_y))]

        def _fit(self):
            """Train the model, and predict on train dataset and test dataset.
            """        
            for i in range(self.n_estimators):
                # Fit a classifier with the specific weights
                self.weak_clf.fit(self.train_X, self.train_y, sample_weight=self.w)
                pred_train_i = self.weak_clf.predict(self.train_X)
                pred_test_i = self.weak_clf.predict(self.test_X)

                # indicate which sample predict incorrectly, miss[0]==1 indicates that the first sample predict incorrectly
                miss = [int(x) for x in (pred_train_i!=self.train_y)]
                erro_m = np.dot(self.w, miss)

                alpha_m = 0.5 * np.log((1-erro_m) / erro_m)
                # new weights
                miss2 = [x if x==1 else -1 for x in miss]

                self.w = np.multiply(self.w, np.exp([float(x) * alpha_m for x in miss2]))
                self.w = self.w / sum(self.w)

                # Add to prediction
                pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
                pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]

                self.pred_train = self.pred_train + np.multiply(alpha_m, pred_train_i)
                self.pred_test = self.pred_test + np.multiply(alpha_m, pred_test_i)

        def _accuracy(self):
            """Calculate the accuracy of the model.

            Returns:
            ---
            train_accu : float
                The accuracy on train dataset.

            test_accu : float
                The accuracy on test dataset.
            """        
            # make the label of positive 1
            _pred_train = (self.pred_train > 0) * 1
            _pred_test = (self.pred_test > 0) * 1
            
            train_accu = sum(_pred_train == self.train_y) / len(self.train_y)
            test_accu = sum(_pred_test == self.test_y) / len(self.test_y)

            return train_accu, test_accu

        def _recall(self):
            """Calculate the recall of the model.

            Returns:
            ---
            train_reca : float
                The recall on train dataset.

            test_reca : float
                The recall on test dataset.
            """    
            # make the label of positive 1
            _pred_train = (self.pred_train > 0) * 1
            _pred_test = (self.pred_test > 0) * 1

            # count samples whose label is positive
            train_positive = np.count_nonzero(self.train_y+1)
            test_positive = np.count_nonzero(self.test_y+1)
            
            train_reca = sum(_pred_train == self.train_y) / train_positive
            test_reca = sum(_pred_test == self.test_y) / test_positive
            
            return train_reca, test_reca

        def _f1(self):
            """Calculate the f1 of the model.

            Returns:
            ---
            train_f1 : float

            test_f1 : float
            """        
            train_accu, test_accu = self._accuracy()
            train_reca, test_reca = self._recall()
            if train_accu==0 and train_reca==0:
                train_f1 = 0.0
            else:
                train_f1 = 2*train_accu*train_reca / (train_accu+train_reca)
            
            if test_accu==0 and test_reca==0:
                test_f1 = 0.0
            else:
                test_f1 = 2*test_accu*test_reca / (test_accu+test_reca)
            return train_f1, test_f1

    data = pd.read_csv(dataFile_path)
    _train_X, _train_y, _test_X, _test_y = Split_Data_Set(data, TEST_PROPORTION)

    train_X, test_X = Vectorize_X(_train_X, _test_X, 'TFIDF')

    for k in range(NUM_CLASSES):
        train_y, test_y = Vectorize_y(_train_y, _test_y, k)
        weak_clf = DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5)
        my_model = My_AdaBoost_Classifier(weak_clf, N_ESTIMATORS, train_X, train_y, test_X, test_y)
        my_model._fit()
        train_accu, test_accu = my_model._accuracy()
        train_f1, test_f1 = my_model._f1()
        print(f'category_{k}, {CATEGORIES_[k]} >> train accu:', train_accu, 'train f1:', train_f1)
        print(f'category_{k}, {CATEGORIES_[k]} >> test accu:', test_accu, 'test f1:', test_f1)
        print('-'*50)


def Sklearn_AdaBoost():
    """Use the sklearn libary AdaBoost model to predict. Convert the multi class problem to
    binary class problem.
    """    
    data = pd.read_csv(dataFile_path)
    _train_X, _train_y, _test_X, _test_y = Split_Data_Set(data, TEST_PROPORTION)

    train_X, test_X = Vectorize_X(_train_X, _test_X, 'TFIDF')

    for k in range(NUM_CLASSES):
        train_y, test_y = Vectorize_y(_train_y, _test_y, k)
        # define the AdaBoost model, which uses decision tree as weak classfier.
        model = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
            algorithm="SAMME", n_estimators=10, learning_rate=0.8)

        # train the model
        clf = model.fit(train_X, train_y)

        print(f'category_{k}, {CATEGORIES_[k]} >> train score:', clf.score(train_X, train_y))
        print(f'category_{k}, {CATEGORIES_[k]} >> test score:', clf.score(test_X, test_y))
        print('-'*50)


def Multiclass_AdaBoost():
    """Use sklearn libary OneVsRestClassifier model, to solve the problem as an entirety, without
    resorting to binarize the problem.
    """    
    data = pd.read_csv(dataFile_path)
    _train_X, _train_y, _test_X, _test_y = Split_Data_Set(data, TEST_PROPORTION)

    train_X, test_X = Vectorize_X(_train_X, _test_X, 'TFIDF')
    train_y, test_y = Multiclass_Label(_train_y, _test_y)

    print('begin training.')
    weak_clf = DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5)
    model = OneVsRestClassifier(AdaBoostClassifier(weak_clf, algorithm="SAMME", n_estimators=10, learning_rate=0.8))
    clf = model.fit(train_X, train_y)
    print('finish training.')

    print('train score:', clf.score(train_X, train_y))
    print('test score:', clf.score(test_X, test_y))



if __name__ == "__main__":
    # My_AdaBoost()
    Sklearn_AdaBoost()
    # Multiclass_AdaBoost()

