import numpy as np
import sklearn
import sklearn.svm, sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from h5py import File
from pylab import *

def trainClassifier(file, fSample):
    erpTarget, erpStimulus, cat, labels = binarizeData(file, fSample)
    model = LogReg(cat, labels)
    return model

def LogReg(data , events):
    lr = LogisticRegression()
    lr.fit(data, events)
    return lr



def SVM(data, events, type = 'target'):
    idxOfType = np.where(events[:, 0] == type)[0]
    eventType = events[idxOfType, :]
    dataType = data[idxOfType, :, : ]

    reshapedDataType = dataType.reshape(dataType.shape[0], dataType.shape[1]* dataType.shape[2])


    uniqueLabels = sorted(list(set(eventType[:,1])))
    label_to_int = dict((l, i) for i, l in enumerate(uniqueLabels))
    int_to_char = dict((i, l) for i, l in enumerate(uniqueLabels))
    convertedLabels = []
    for label in eventType[:,1]:
    	convertedLabels.append([label_to_int[label]])

    tmp = np.array(convertedLabels)

    test = MultiLabelBinarizer().fit_transform(tmp)

    # print(test)
    # print(eventType[:,1])
    from sklearn import svm
    model = OneVsRestClassifier(svm.SVC(class_weight='balanced', probability = 1))
    # print(eventType[:,1].shape)
    model.fit(reshapedDataType, test)
    # returns trained model
    return model, reshapedDataType



def binarizeData(file, fSample):
    with File(file, 'r') as f:
        events      = f['events'].value
        data        = f['processedData'].value

    # make a mapper for the classifier
    labels           = np.zeros(2)
    labels[0]        = 1

    # get the indices corresponding to the conditions
    # events is ordered as [0] = type, [1] = value
    targets         = np.where(events[:, 0] == 'target')[0]
    stimuli         = np.where(events[:, 0] == 'stimulus')[0]


    # make dataset for clasifier
    timeAverage     = np.mean(data, 1)
    avgTarget       = np.mean(timeAverage[targets,:],0)
    avgStimulus     = np.mean(timeAverage[stimuli, :], 0)
    cat             = np.vstack((avgTarget, avgStimulus))


    # plot the erps of the classes
    tData           = data[targets, :, :]
    sData           = data[stimuli, :, :]

    erpTarget       = np.mean(tData, 0)
    erpStimulus     = np.mean(sData, 0)

    # plot classifier erps
    fig, axs = subplots(erpTarget.shape[1], sharex = 'all')
    time = np.arange(0, data.shape[1]) / float(fSample)
    for i, ax in enumerate(axs):
        # target
        ax.plot(time, erpTarget[:, i],\
        label   = 'class 1 ({0})'.format(tData.shape[0]))

        # stimulus
        ax.plot(time, erpStimulus[:, i],\
        label   = 'class 0 ({0})'.format(sData.shape[0]))

        # formatting
        ax.legend(loc = 0)
        ax.set_xlabel('Time [s]')
    fig.tight_layout()
    # save the figure
    savefig('../Figures/Classifier_output.png')
    return erpTarget, erpStimulus, cat, labels


if __name__ == '__main__':
    from h5py import File
    with File('../Data/calibration_subject_4_LAB.hdf5') as f:
        for i in f: print(i)
        data = f['processedData'].value
        cap  = f['cap'].value
        events = f['events'].value
    model, reshapedData = SVM(data, events)
    out = model.predict_proba(reshapedData)
    print(out)
    fig, ax = subplots()
    print(cap)


    # print(model.predict_proba(reshapedData))
