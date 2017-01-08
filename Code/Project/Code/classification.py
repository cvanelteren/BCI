import numpy as np
import sklearn
import sklearn.svm, sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from h5py import File
from pylab import subplots, savefig

def trainClassifier(file, fSample):
    erpTarget, erpStimulus, cat, labels = binarizeData(file, fSample)
    model = LogReg(cat, labels)
    return model

def LogReg(data , events):
    lr = LogisticRegression()
    lr.fit(data, events)
    return lr



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
