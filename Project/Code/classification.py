import numpy as np
import sklearn
import sklearn.svm, sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from h5py import File
import pickle
from pylab import *


def trainClassifier(file, fSample):
    erpTarget, erpStimulus, cat, labels = binarizeData(file, fSample)
    model = LogReg(cat, labels)
    return model


def LogReg(data, events):
    lr = LogisticRegression()
    lr.fit(data, events)
    return lr


def SVM(data, events, type = 'target', string='default'):
    idxOfType       = np.where(events[:, 0] == type)[0]
    eventType       = events[idxOfType, :]
    dataType        = data[idxOfType, :, :]

    reshapedDataType = dataType.reshape(dataType.shape[0],\
                                        dataType.shape[1] *
                                        dataType.shape[2])

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


    # print(test)
    # print(eventType[:,1])
    from sklearn import svm
    model = OneVsRestClassifier(
        svm.SVC(class_weight='balanced', probability=1))
    # print(eventType[:,1].shape)
    model.fit(reshapedDataType, test)
    # returns trained model
    return model, reshapedDataType, eventType




if __name__ == '__main__':
    from h5py import File
    from preproc import  stdPreproc
    with File('../Data/calibration_subject_5.hdf5') as f:
        for i in f:
            print(i)
        data = f['processedData'].value
        rawData = f['rawData'].value
        cap = f['cap'].value
        events = f['events'].value

    # model, reshapedData, tmp = SVM(data, events)
    modelERN, reshapedData, eventTarget = SVM(data, events, type='feedback')
    # tmp = np.array(reshapedData[0,:], ndmin)
    out = modelERN.predict_proba(reshapedData[[0],:])


    print(out) # eventTarget[:, 1])
    idx = 20
    # print(out[:idx], tmp[:idx,1])
    fig, ax = subplots()

    # print(model.predict_proba(reshapedData))
