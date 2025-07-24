import numpy as np
from sklearn.naive_bayes import BernoulliNB
# from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def bayes(train_x, train_y, test_x):
    # bnb = BernoulliNB()
    bnb = GaussianNB()
    bnb.fit(train_x, train_y)
    predict_y = bnb.predict(test_x)

    return predict_y
