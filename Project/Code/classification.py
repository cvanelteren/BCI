import numpy as np
import sklearn
import sklearn.svm, sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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
    print(test.shape)

    from sklearn import svm
    # model = OneVsRestClassifier(svm.SVC(probability = 1))
    # cw = {1: 1/4., 2: 1/4.,3:1/8.,4:1/4.}
    model  = OneVsRestClassifier(svm.SVC(class_weight = 'balanced', kernel = 'sigmoid', probability=1))
    # print(eventType[:,1].shape)
    model.fit(reshapedDataType, test)
    # returns trained modelocData, ev
    return model, reshapedDataType, test




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

    # tmp = sklearn.preprocessing.normalize(rawData, axis = 1)
    # plotERP()
    restCondition = np.where(events == 'rest')[1]
    useIdx  = len(restCondition) / 3
    np.random.shuffle(restCondition)
    restCondition = restCondition[useIdx:]
    useThese = np.zeros((procData.shape[0]))
    useThese[restCondition] = 1
    modelIM, reshapedData, target = SVM(procData[useThese == 0 , :], events[useThese==0,:], type = 'target',string='im')
    modelERN, rd , rt          = SVM(rawData, events, type = 'feedback',string='ern')
    print(modelIM)
    # modelERN, reshapedData, eventTarget = SVM(data, events, type='feedback')
    # modelIM, rehsapdeData, eventTarget = SVM(rawData, events, type = 'target')
    # tmp = np.array(reshapedData[0,:], ndmin)
    out = modelIM.predict(reshapedData[:,:])
    tmp = events[useThese==0,:]

    print(modelIM.score(reshapedData, target))
    print(modelERN.score(rd, rt))
    #print(out[:5,:])
    #print(tmp[tmp[:,0] == 'target',:][:, :5])
