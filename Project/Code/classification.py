from    __future__  import print_function, division
from    h5py        import File
from    visualize   import plotConfusionMatrix
from    preproc     import stdPreproc
from    pylab       import *

import numpy as np
import sklearn
import sklearn.svm # for some reason it needs to be imported like this
import pickle, multiprocessing



def SVM(data, events, numCrossFold = 10, fft = 0, cs  = np.linspace(.01, 10, 30)):
    '''
    Returns the optimal SVM for the input data using k-fold cross validation.
    The SVM is weighted according to the the size of the class labels in events
    Inputs:
          - data : trials x time x channels

          - events : type x class types

          - numCrossFold : number of cross validation

          - fft  : used for the imagined movement condition, i.e. classifier is
          trained on the power of the data

          - cs   : input for the grid search, C controls the margin of error for the fitted
          lines of the SVM

    This function will also output performance of the optimal method found using grid search
    returns trained SVM on the data
    '''

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


    gs = sklearn.model_selection.GridSearchCV                        # init grid search object
    parameters = {'kernel':('linear','sigmoid','rbf'),               # parameters grid search
                    'C':cs}



    # print('Class weights: \n\t : ', classWeights)
    model = sklearn.svm.SVC(class_weight = 'balanced', probability = 1)
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
        XTrain, YTrain = data[train_index, ...], events[train_index, 1] # define train set
        XTest, YTest = data[test_index, ...], events[test_index, 1]     # define test set

        model.fit(XTrain, YTrain)                                       # fit the model

        yPred = model.predict(XTest)                                    # predict on test set
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

    fig = plotConfusionMatrix(conf, classWeights.keys())
    # print('Confusion matrix : \n', sklearn.metrics.confusion_matrix(events[:,1], pred))
    print('Mean accuracy {0} +-{1}'.format(np.mean(accuracy), np.std(accuracy)))
    # print(accuracy)
    model.fit(data, events[:,1])
    return model, fig







if __name__ == '__main__':
    '''
    If this is run separately, it will train classifier on input data
    Please run this from a terminal
    '''

    from h5py import File
    from preproc import  stdPreproc
    import sklearn, sklearn.preprocessing
    from systemHelper import enterSubjectNumber
    # file = enterSubjectNumber(10412)
    subjectNumber = 4
    file = '../Data/calibration_subject_{0}.hdf5'.format(subjectNumber) # uncomment for mockdata
    # file = '../Data/crossValSetExampleJason.hdf5'   # exmaple data from matlab/offline  for testing if our pipeline worked as it should
    with File(file) as f:
        print('File contents:\n')
        for i in f:
            print(i)
        rawDataIM    = f['rawData/IM'].value
        rawDataERN  = f['rawData/ERN'].value
        # procDataIM   =  f['procData/IM'].value
        # procDataERN  = f['procData/ERN'].value
        eventsIM     = f['events/IM'].value
        eventsERN    = f['events/ERN'].value
    import sklearn


    # preprocessing is done because for the earlier subjects we were figuring out
    # the preprocessing pipeline, hence to be sure everything is the same; run the preprocessing.

    procDataIM, _  = stdPreproc(rawDataIM, [8,20], 250)
    procDataERN, _ = stdPreproc(rawDataERN, [1,40], 250)

    print('Training IM classifier')
    model, fig    = SVM(procDataIM, eventsIM, fft = 1)
    fig.savefig('../Figures/Confusion_IM_subject_{0}.pdf'.format(subjectNumber))

    # print('Training FRN classifier')
    # model, fig    = SVM(procDataERN, eventsERN)
    # fig.savefig('../Figures/Confusion_IM_subject_{0}.pdf'.format(subjectNumber))
