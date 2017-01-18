import numpy as np
import sklearn
import sklearn.svm, sklearn.linear_model
from sklearn.linear_model import LogisticRegression

def linear_svm(data, events,  C = 1):
    svm = sklearn.svm.SVC(kernel = 'linear', C = C)
    svm.fit(data, events)
    return svm

def LogReg(data , events):
    lr = LogisticRegression()
    lr.fit(data, events)
    return lr
