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





def SVM(data, events, numCrossFold = 10, fft = 0):
    # numCrossFold = data.shape[0]-1
    uniques                =  np.unique(events[:,1])
    # weigh the class weights according to occurence
    # to obtain the class weight, first find the maximum amongst classes
    # then divide by the number of cases per class
    classWeights   = {}
    normalizer     = 0

    for idx, unique in enumerate(uniques):
        sizeOfCond  = len(np.where(events[:, 1] == unique)[0])
        if idx == 0 :
            # perform forward sweep to find the max used for computing ratio
            for i in uniques:
                tmp = len(np.where(events[:, 1] == i)[0])
                # print(tmp, normalizer)
                if tmp > normalizer:
                    normalizer = tmp
        # set the class weights
        classWeights[unique] = normalizer /  sizeOfCond


    # concatenate channels x time
    # power over features
    print('Data', data.shape)
    if fft:
        data = abs(np.fft.fft(data.reshape(data.shape[0], -1), axis = 0))**2
    else:
        data = data.reshape(data.shape[0], -1)
    cs  = np.linspace(.01, 10, 30)

    gs = sklearn.model_selection.GridSearchCV                        # init grid search object
    parameters = {'kernel':('linear','rbf'),              # parameters grid search
                    'C':cs}


    # classWeights = 'balanced'
    print('Class weights: \n\t : ', classWeights)
    model = sklearn.svm.SVC(class_weight = classWeights, probability = 1)
    cv    = gs(model,
               parameters,
               cv = numCrossFold,
               verbose = 1,
               n_jobs = multiprocessing.cpu_count(),
               scoring = 'recall_micro')
    cv.fit(data, events[:,1])
    model = cv.best_estimator_

    pred = model.predict(data)



    print('--' * 70)
    print('Best parameters', cv.best_params_)

    # get the model accuracy for the optimal model defined above
    kf = sklearn.model_selection.KFold(n_splits= numCrossFold)

    # define the confusion matrix
    tmp = len(classWeights.keys())
    conf = np.zeros( (tmp, tmp) )
    accuracy = []
    # for all kfolds compute the confusion matrix
    for train_index, test_index in kf.split(data):
        XTrain, YTrain = data[train_index, ...], events[train_index, 1]
        XTest, YTest = data[test_index, ...], events[test_index, 1]

        model.fit(XTrain, YTrain)

        yPred = model.predict(XTest)
        # in case not all labels are predicted, construct the conf matrix per
        # fold
        tmp = zeros(conf.shape)
        for pred, tar in zip(yPred, YTest):
            for idx, label in enumerate(classWeights.keys()):
                if pred == label:
                    rower = idx
                if tar == label:
                    coller = idx
            tmp[coller, rower] += 1
        # print(sklearn.metrics.classification_report(YTest, yPred, target_names = classWeights.keys()))

        conf += tmp
        accuracy.append(sum(tmp.diagonal()) / len(YTest))

    print('Confusion matrix : \n', conf)
    # print('Confusion matrix : \n', sklearn.metrics.confusion_matrix(events[:,1], pred))
    print('Mean accuracy {0} +-{1}'.format(np.mean(accuracy), np.std(accuracy)))
    model.fit(data, events[:,1])
    return model







if __name__ == '__main__':
    from h5py import File
    from preproc import  stdPreproc
    import sklearn, sklearn.preprocessing
    with File('../Data/calibration_subject_MOCK_3.hdf5') as f:
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
    data    = [procDataIM, procDataERN]
    events  = [eventsIM, eventsERN]

    # print(data)

    model = SVM(procDataIM, eventsIM, fft = 1)
    model = SVM(procDataERN, eventsERN)
