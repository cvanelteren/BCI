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
import multiprocessing
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





def SVM(data, events, numCrossFold = 2, fft = 0):
    uniques                =  np.unique(events[:,1])
    # weigh the class weights according to occurence
    # to obtain the class weight, first find the maximum amongst classes
    # then divide by the number of cases per class
    classWeights   = {}
    normalizer     = 0

    for idx, unique in enumerate(uniques):
        if idx == 0 :
            # perform forward sweep to find the max used for computing ratio
            for i in uniques:
                sizeOfCond  = len(np.where(events[:, 1] == i)[0])
                if sizeOfCond > normalizer:
                    normalizer = sizeOfCond
            classWeights[unique] = normalizer /  len(np.where(events[:, 1] == i)[0])
        else:
            classWeights[unique] = normalizer /  len(np.where(events[:, 1] == i)[0])

    print('Class weights: \n\t : ', classWeights)
    # concatenate channels x time
    # power over features
    print('Data', data.shape)
    if fft:
        data = abs(np.fft(data.reshape(data.shape[0], -1), axis = 0))**2
    else:
        data = data.reshape(data.shape[0], -1)

    gs = sklearn.model_selection.GridSearchCV                        # init grid search object
    parameters = {'kernel':('linear', 'sigmoid','rbf'),              # parameters grid search
                    'C':np.linspace(.01, 3,20)}
    # cw         = {'left hand':5, 'right hand':5, 'feet':5, 'rest':1} # class weights : deprecated!


    model = sklearn.svm.SVC(class_weight = classWeights, probability = 1)       # probabilities are odd --> take argmin or argmax 1 - p
    cv    = gs(model, parameters, cv = numCrossFold, verbose = 1, n_jobs = multiprocessing.cpu_count())
    cv.fit(data, events[:,1])
    model = cv.best_estimator_
    model.fit(data, events[:,1])
    print('--' * 50)
    print('Best parameters', cv.best_params_)
    print('Mean cross validation test score {0} +- {1}'.format(
    np.mean(cv.cv_results_['mean_test_score']),
    np.std(cv.cv_results_['std_test_score']) ))

    return model
    # # obtain the optimal model
    # IMGridSearch = gs(modelIM, parameters, cv = numCrossFold, verbose = 1)
    # IMGridSearch.fit(*datasetIM)
    # modelIM = IMGridSearch.best_estimator_                                     # optimal model action
    # # print('Mean test validation score\n', c.cv_results_['mean_test_score'])
    #
    #
    # print('Training ERN classifier')
    # # slightly edit the model above for ERN:
    # modelERN = sklearn.svm.SVC(class_weight = classWeightingERN,
    #                         probability = True)
    #
    # ERNGridSearch                       = gs(modelERN, parameters, cv = numCrossFold, verbose = 1)
    # ERNGridSearch.fit(*datasetERN)
    #
    # modelERN                            = ERNGridSearch.best_estimator_
    # print('*' * 32)
    # print('Best Parameters for IM : \n\t', IMGridSearch.best_params_)
    # print('Mean test score of CV {0} +- {1}'.format(
    #                                         np.mean(IMGridSearch.cv_results_['mean_test_score']),
    #                                         np.std(IMGridSearch.cv_results_['std_test_score']) )
    #                                         )
    # print('*' * 32)
    # print('Best Parameters for ERN : \n\t', ERNGridSearch.best_params_)
    # print('Mean test score of CV {0} +- {1}'.format(
    #                                         np.mean(ERNGridSearch.cv_results_['mean_test_score']),
    #                                         np.std(ERNGridSearch.cv_results_['std_test_score']))
    #                                         )
    # print('*' * 32)
    # print('Done with training')
    #
    # # fit the models (ready for use)
    # modelIM.fit(*datasetIM)
    # modelERN.fit(*datasetERN)
    # return modelIM, modelERN






if __name__ == '__main__':
    from h5py import File
    from preproc import  stdPreproc
    import sklearn, sklearn.preprocessing
    with File('../Data/calibration_subject_MOCK_22.hdf5') as f:
        for i in f:
            print(i)


        procDataIM =  f['procData/IM'].value
        procDataERN = f['procData/ERN'].value
        eventsIM    = f['events/IM'].value
        eventsERN    = f['events/ERN'].value
        # # assert 0
        # rawData = f['rawData'].value
        # procData = f['processedData'].value
        # cap = f['cap'].value
        # events = f['events'].value
    # kcrossVal(procData, events)
    import sklearn
    data = [procDataIM, procDataERN]
    events = [eventsIM, eventsERN]
    print(procDataERN.shape)
    # print(data)

    model = SVM(procDataIM, eventsIM)
    model = SVM(procDataERN, eventsERN)
