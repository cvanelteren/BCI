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


def SVM(data, events, c, type = 'target', string='default'):
    idxOfType       = np.where(events[:, 0] == type)[0]
    eventType       = events[idxOfType, :]
    dataType        = data[idxOfType, :, :]

    reshapedDataType = dataType.reshape(dataType.shape[0],-1)
    # print(dataType.shape, eventType)

    uniqueLabels         = sorted(list(set(eventType[:, 1])))
    label_to_int         = dict((l, i) for i, l in enumerate(uniqueLabels))
    int_to_label          = dict((i, l) for i, l in enumerate(uniqueLabels))

    filepath_l2i = 'l2i_' + string +'.pkl'
    filepath_i2l = 'i2l_' + string +'.pkl'
    pickle.dump(label_to_int,open(filepath_l2i,'wb'))
    pickle.dump(int_to_label,open(filepath_i2l,'wb'))

    convertedLabels      = []
    for label in eventType[:, 1]:
        convertedLabels.append([label_to_int[label]])

    tmp = np.array(convertedLabels)

    test = MultiLabelBinarizer().fit_transform(tmp)

    from sklearn import svm
    # model = OneVsRestClassifier(svm.SVC(probability = 1))
    # cw = {1: 1/4., 2: 1/4.,3:1/8.,4:1/4.}
    model  = OneVsRestClassifier(\
    svm.SVC(C = c, class_weight = 'balanced', kernel = 'rbf', probability=1))
    # print(eventType[:,1].shape)
    model.fit(reshapedDataType, test)
    # returns trained modelocData, ev
    return model, reshapedDataType, test

def kcrossVal(data, events, CMax = 1, N = 10, threshold = 1e-30):
    '''
    Returns optimals model for data:
    '''

    from sklearn.model_selection import KFold, cross_val_score
    from sklearn import preprocessing
    from sklearn import svm

    c = linspace(0.01, CMax, 1000)
    # print(c)
    # find indices of the evenst
    ernEvent = np.where(events[:,0] == 'feedback')[0]
    actEvent = np.where(events[:,0] == 'target')[0]

    tmpEventAct = events[actEvent, :]
    tmpEventERN = events[ernEvent, :]

    tmpData   = data[actEvent,  :]
    tmpDataERN = data[ernEvent, :]

    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import GridSearchCV
    binEvents = preprocessing.LabelBinarizer().fit_transform(tmpEventAct[:,1])
    binEventsERN = preprocessing.LabelBinarizer().fit_transform(tmpEventERN[:,1])
    idx = np.where(binEventsERN == 1)[0]
    jdx = np.where(binEventsERN == 0)[0]
    tmp = zeros((binEventsERN.shape[0],2))
    tmp[idx, 0] = 1
    tmp[jdx, 1] = 1

    binEventsERN = tmp


    modelsReturn = []
    models = []
    cNew = cOld = 0
    from sklearn.model_selection import LeaveOneOut
    print('Training Action classifier')
    for ci in c:
        model = OneVsRestClassifier(svm.SVC(C = ci, kernel = 'rbf', class_weight ='balanced'))
        rD = tmpData.reshape(tmpData.shape[0],-1)
        scores = cross_val_score(model, rD, binEvents, cv = N,n_jobs =4)
        print(scores.mean(), scores.std())
        cNew = scores.mean()
        # print()
        if (cNew - cOld)**2 > threshold:
            cOld = cNew
            models.append(model)
        else:
            modelsReturn.append(models[-1])
            break
    print(len(models))
    modelsReturn[-1].fit(tmpData.reshape(tmpData.shape[0],-1), binEvents)
    print(modelsReturn[-1].score(tmpData.reshape(tmpData.shape[0],-1), binEvents))
    print(modelsReturn[-1].predict(tmpData.reshape(tmpData.shape[0],-1)) == binEvents)
    print(cNew,'reg strength', ci)
    print('Training ERN classifier')
    models = []
    cNew = cOld = 0
    for ci in c:
        model = OneVsRestClassifier(svm.SVC(C = ci, kernel = 'rbf', class_weight ='balanced'))
        rD = tmpDataERN.reshape(tmpDataERN.shape[0],-1)
        scores = cross_val_score(model, rD, binEventsERN, cv = N)
        print(scores.mean(), scores.std())
        cNew =  scores.mean()
        if (cNew - cOld)**2 > threshold:
            cOld = cNew
            models.append(model)
        else:
            modelsReturn.append(models[-1])
            break
    # modelsReturn[-1].fit(tmpDataERN.reshape(tmpDataERN.shape[0],-1), binEventsERN)
    # print(modelsReturn[-1].score(tmpData.reshape(tmpDataERN.shape[0],-1), binEventsERN))
    print(cNew,'reg strength', ci)
    return modelsReturn






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
    kcrossVal(procData, events)
