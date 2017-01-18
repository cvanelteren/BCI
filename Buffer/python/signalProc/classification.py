import numpy as np
import sklearn
import sklearn.svm

def linear_svm(data, events,  C = 1):
    svm = sklearn.svm.SVC(kernel = 'linear', C = C)
    svm.fit(data, events)
    return svm
