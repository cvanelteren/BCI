from __future__ import print_function
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

#
# def SVM(data, events, c, type = 'target', string='default'):
#     idxOfType       = np.where(events[:, 0] == type)[0]
#     eventType       = events[idxOfType, :]
#     dataType        = data[idxOfType, :, :]
#
#     reshapedDataType = dataType.reshape(dataType.shape[0],-1)
#     # print(dataType.shape, eventType)
#
#     uniqueLabels         = sorted(list(set(eventType[:, 1])))
#     label_to_int         = dict((l, i) for i, l in enumerate(uniqueLabels))
#     int_to_label          = dict((i, l) for i, l in enumerate(uniqueLabels))
#
#     filepath_l2i = 'l2i_' + string +'.pkl'
#     filepath_i2l = 'i2l_' + string +'.pkl'
#     pickle.dump(label_to_int,open(filepath_l2i,'wb'))
#     pickle.dump(int_to_label,open(filepath_i2l,'wb'))
#
#     convertedLabels      = []
#     for label in eventType[:, 1]:
#         convertedLabels.append([label_to_int[label]])
#
#     tmp = np.array(convertedLabels)
#
#     test = MultiLabelBinarizer().fit_transform(tmp)
#
#     from sklearn import svm
#     # model = OneVsRestClassifier(svm.SVC(probability = 1))
#     # cw = {1: 1/4., 2: 1/4.,3:1/8.,4:1/4.}
#     model  = OneVsRestClassifier(\
#     svm.SVC(C = c, class_weight = 'balanced', kernel = 'rbf', probability=1))
#     # print(eventType[:,1].shape)
#     model.fit(reshapedDataType, test)
#     # returns trained modelocData, ev
#     return model, reshapedDataType, test



def SVM(data, events, numCrossFold = 2):
    import sklearn
    targetEventIdx      = np.where(events[:,0] == 'target')[0]
    feedbackEventsIdx   = np.where(events[:,0] == 'feedback')[0]

    reshapeData = data.reshape(data.shape[0],-1)
    datasetIM   = [reshapeData[targetEventIdx, :],
                    events[targetEventIdx, 1]]     # input for IM model

    datasetERN  = [reshapeData[feedbackEventsIdx,:],
                   events[feedbackEventsIdx,  1]] # input for ERN model

    print('Training IM classifier')
    gs = sklearn.model_selection.GridSearchCV                        # init grid search object
    parameters = {'kernel':('linear', 'sigmoid','rbf'),              # parameters grid search
                    'C':np.linspace(.01,3,20)}
    cw         = {'left hand':3, 'right hand':3, 'feet':3, 'rest':1} # class weights
    model = sklearn.svm.SVC(class_weight= cw, probability = 1)       # probabilities are odd --> take argmin or argmax 1 - p

    # obtain the optimal model
    c = gs(model, parameters, cv = numCrossFold, verbose = 1, n_jobs = 8)
    c.fit(*datasetIM)
    print(np.max(c.cv_results_['mean_test_score']))
    modelIM = c.best_estimator_                                     # optimal model action
    print('Mean test validation score\n', c.cv_results_['mean_test_score'])


    print('Training ERN classifier')
    # slightly edit the model above for ERN:
    model = sklearn.svm.SVC(class_weight={'positive':1,'negative': 3},
                            probability = True)

    c                       = gs(model, parameters, cv = numCrossFold, verbose = 1, n_jobs= 8)
    c.fit(*datasetERN)
    print(np.max(c.cv_results_['mean_test_score']))
    modelERN                = c.best_estimator_
    print(modelERN)
    print(modelIM)             # optimal model ERN
    print('Mean test validation score\n', c.cv_results_['mean_test_score'])

    # modelERN.fit(*datasetERN)
    modelIM.fit(*datasetIM)

    return modelIM, modelERN






if __name__ == '__main__':
    from h5py import File
    from preproc import  stdPreproc
    import sklearn, sklearn.preprocessing
    with File('../Data/calibration_subject_MOCK_6.hdf5') as f:
        for i in f:
            print(i)

        # assert 0
        rawData = f['rawData'].value
        procData = f['processedData'].value
        cap = f['cap'].value
        events = f['events'].value
    # kcrossVal(procData, events)
    import sklearn
    SVM(procData, events)
