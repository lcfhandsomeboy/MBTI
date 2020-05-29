#coding:utf-8
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize


dataFile_path = './mbti.csv'

# label each personality
CATEGORIES = {'INTJ':0, 'INTP':1, 'INFJ':2, 'INFP':3, 'ISTJ':4, 'ISTP':5, 'ISFJ':6, 'ISFP':7,
    'ENTJ':8, 'ENTP':9, 'ENFJ':10, 'ENFP':11, 'ESTJ':12, 'ESTP':13, 'ESFJ':14, 'ESFP':15}
CATEGORIES_ = {val:key for key,val in CATEGORIES.items()}
NUM_CLASSES = len(CATEGORIES)


def Draw_word_cloud(data):
    """User word cloud to illustrate that the frequency of post words is relevant to the personality. 
    - Here we use Introversion vers Extroversion as the example.

    Arguments:
    ----------
    data : pandas df
    """    
    type_quote = data.groupby('type').sum()
    # type_quote.info()
    # print(type_quote.index)

    e_posts = ''
    i_posts = ''
    for _type in type_quote.index:
        if 'E' in _type:
            e_posts += type_quote.loc[_type].posts
        else:
            i_posts += type_quote.loc[_type].posts

    # Generate wordcloud 
    stopwords = set(STOPWORDS)
    stopwords.add("think")
    stopwords.add("people")
    stopwords.add("thing")
    my_wordcloud = WordCloud(width=800, height=800, stopwords=stopwords, background_color='white')
    # Introvert 
    my_wordcloud_i = my_wordcloud.generate(i_posts)
    plt.subplots(figsize = (15,15))
    plt.imshow(my_wordcloud_i)
    plt.axis("off")
    plt.title('Introvert', fontsize = 30)
    plt.show()
    #Extrovert 
    my_wordcloud_e = my_wordcloud.generate(e_posts)
    plt.subplots(figsize = (15,15))
    plt.imshow(my_wordcloud_e)
    plt.axis("off")
    plt.title('Extrovert', fontsize = 30)
    plt.show()


def Split_Data_Set(data, test_proportion):
    """Split the original dataset into train and test. 

    Arguments:
    ---
    data : pandas df
        original dataset

    test_proportion : float
        the proportion of test dataset

    Returns:
    ---
    train_X, train_y, test_X, test_y : ndarray
    """    
    train_X, test_X, train_y, test_y = train_test_split(data['posts'], data['type'], test_size=test_proportion)
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    return train_X, train_y, test_X, test_y


def Vectorize_X(train_X, test_X, method):
    """Vectorize the traing input and test input.

    Arguments:
    ---
    train_X : csr_martix
        training input dataset  to be vectorized

    test_X : csr_matrix
        test input dataset to be vectorized

    method : string
        TFIDF, word2vec

    Return:
    ---
    train_X : ndarray

    test_X : ndarray
    """    
    if method == 'TFIDF':
        # drop the stop words
        tfidf = TfidfVectorizer(stop_words='english')
        # fit/train the tfidf matrix by train data
        tfidf.fit(train_X)
        train_X = tfidf.transform(train_X)
        test_X = tfidf.transform(test_X)

        # transform the csr_matrix to ndarray
        # train_X = train_X.A
        # test_X = test_X.A
        return train_X, test_X
    else:
        if os.path.exists('./word2vec.model'):
            model = Word2Vec.load('./word2vec.model')
        else:
            model = Word2Vec(train_X, sg=0, size=100, window=5, min_count=5, negative=3, sample=0.001, hs=1, workers=4)
            model.save('./word2vec.model')
        print(model)


def Vectorize_y(train_y, test_y, curr_cate):
    """There are total 16 classes in the MBTI dataset, so we need to preprocess the labels.
    Binary the dataset.

    Arguments:
    ---
    train_y : ndarray

    test_y : ndarray

    curr_cate : int
        0-15, figure out the current basic category, convert the label of basic 
        category to 1, and convert the label of other categories to -1.

    Returns:
    ---
        train_y : ndarray
        test_y : ndarray
    """    
    temp = np.zeros(len(train_y))
    for i in range(len(train_y)):
        temp[i] = 1 if train_y[i]==CATEGORIES_[curr_cate] else -1
    train_y = np.array(temp, dtype=np.int)

    temp = np.zeros(len(test_y))
    for i in range(len(test_y)):
        temp[i] = 1 if test_y[i]==CATEGORIES_[curr_cate] else -1
    test_y = np.array(temp, dtype=np.int)

    return train_y, test_y


def Multiclass_Label(train_y, test_y):
    """Label each sample with 0-15, convert the original string label to number label.

    Returns:
    ---
        train_y : ndarray

        test_y : ndarray
    """    
    for i in range(len(train_y)):
        train_y[i] = CATEGORIES[train_y[i]]
    
    for i in range(len(test_y)):
        test_y[i] = CATEGORIES[test_y[i]]
    
    train_y = np.array(train_y, dtype=np.int)
    test_y = np.array(test_y, dtype=np.int)
    return train_y, test_y


if __name__ == "__main__":
    data = pd.read_csv(dataFile_path)

    # draw word cloud graph
    Draw_word_cloud(data)