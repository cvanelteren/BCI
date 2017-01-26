from __future__ import print_function, division
import numpy as np

import sklearn
import sklearn.svm, sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from h5py import File
import pickle
from pylab import *
from preproc import stdPreproc

def stupidFct():
    return 42

def trainClassifier(file, fSample):
    erpTarget, erpStimulus, cat, labels = binarizeData(file, fSample)
    model = LogReg(cat, labels)
    return model


def LogReg(data, events):
    lr = LogisticRegression()
    lr.fit(data, events)
    return lr





def SVM(data, events, numCrossFold = 2):
    import sklearn
    targetEventIdx      = np.where(events[:,0] == 'target')[0]
    feedbackEventsIdx   = np.where(events[:,0] == 'feedback')[0]


    uniqueEvents        = np.unique(events[targetEventIdx, 1])

    # weigh the class weights according to occurence
    # to obtain the class weight, first find the maximum amongst classes
    # then divide by the number of cases per class
    cw = {}
    normalizer = 0
    for idx, uniqueEvent in enumerate(uniqueEvents):
        if idx == 0 :
            # perform forward sweep to find the max used for computing ratio
            for tmp in uniqueEvents:
                current = len(np.where(events[:, 1] == tmp)[0])
                if current > normalizer:
                    normalizer = current # check for the max case
            # divide number of max case / number of cases
            cw[uniqueEvent] = normalizer / len(np.where(events[:, 1] == uniqueEvent)[0])
        else:
            cw[uniqueEvent] = normalizer / len(np.where(events[:,1] == uniqueEvent)[0])
    print(cw)
    reshapeData = data.reshape(data.shape[0],-1)
    datasetIM   = [abs(np.fft.fft(reshapeData[targetEventIdx, :], axis = 0))**2,
                    events[targetEventIdx, 1]]     # input for IM model

    datasetERN  = [reshapeData[feedbackEventsIdx,:],
                   events[feedbackEventsIdx,  1]] # input for ERN model

    print('Training IM classifier')
    gs = sklearn.model_selection.GridSearchCV                        # init grid search object
    parameters = {'kernel':('linear', 'sigmoid','rbf'),              # parameters grid search
                    'C':np.linspace(.01,3,20)}
    # cw         = {'left hand':3, 'right hand':3, 'feet':3, 'rest':1} # class weights : deprecated!
    model = sklearn.svm.SVC(class_weight= cw, probability = 1)       # probabilities are odd --> take argmin or argmax 1 - p

    # obtain the optimal model
    IMGridSearch = gs(model, parameters, cv = numCrossFold, verbose = 1, n_jobs = 8)
    IMGridSearch.fit(*datasetIM)
    modelIM = IMGridSearch.best_estimator_                                     # optimal model action
    # print('Mean test validation score\n', c.cv_results_['mean_test_score'])


    print('Training ERN classifier')
    # slightly edit the model above for ERN:
    model = sklearn.svm.SVC(class_weight={'positive':1,'negative': 3},
                            probability = True)

    ERNGridSearch                       = gs(model, parameters, cv = numCrossFold, verbose = 1, n_jobs= 8)
    ERNGridSearch.fit(*datasetERN)

    modelERN                            = ERNGridSearch.best_estimator_
    print('*' * 32)
    print('Best Parameters for IM : \n\t', IMGridSearch.best_params_)
    print('Mean test score of CV {0} +- {1}'.format(
                                            np.mean(IMGridSearch.cv_results_['mean_test_score']),
                                            np.std(IMGridSearch.cv_results_['std_test_score']) )
                                            )
    print('*' * 32)
    print('Best Parameters for ERN : \n\t', ERNGridSearch.best_params_)
    print('Mean test score of CV {0} +- {1}'.format(
                                            np.mean(ERNGridSearch.cv_results_['mean_test_score']),
                                            np.std(ERNGridSearch.cv_results_['std_test_score']))
                                            )
    print('*' * 32)
    print('Done with training')

    # fit the models (ready for use)
    modelIM.fit(*datasetIM)
    modelERN.fit(*datasetERN)
    return modelIM, modelERN






if __name__ == '__main__':
    from h5py import File
    from preproc import  stdPreproc
    import sklearn, sklearn.preprocessing
    with File('../Data/calibration_subject_5.hdf5') as f:
        for i in f:
            print(i)

        # assert 0
        rawData = f['rawData'].value
        procData = f['processedData'].value
        cap = f['cap'].value
        events = f['events'].value
    # kcrossVal(procData, events)
    import sklearn
    a,b = SVM(procData, events)
