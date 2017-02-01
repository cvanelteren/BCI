from __future__ import print_function, division

from pylab import *
import numpy as np
import scipy, scipy.signal, itertools

def plotConfusionMatrix(cm,
                        classes,
                        normalize=False,
                        title = 'Confusion matrix',
                        cmap = cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    colorbar(im, ax = ax)
    tick_marks = np.arange(len(classes))
    xticks(tick_marks, classes, rotation=45)
    yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    tight_layout()
    ylabel('True label')
    xlabel('Predicted label')
    # show()
    return fig

def plotERP(data, events, cap, fSample = 100):
    '''
    Plots the ERP of the data per condition as indicated in events

    Inputs :
            Data = trials x time x channels
            Events = type x Value
    '''

    uniques = np.unique(events[:,1])                # find unique values of events

    # init mainframe and subplots
    nCols = 2 # default value to have at most 2 columns
    nRows = int(ceil(data.shape[-1] / nCols))
    fig, axes = subplots(nrows = nRows,
                         ncols = nCols,
                         sharex =  'all',
                         sharey = 'all')

    mainFrame = fig.add_subplot(111, frameon = 0)
    time = np.arange(0, data.shape[1] * 1/fSample, 1/fSample)

    # for every channel print erp of conditions

    for unique in uniques:
        idx       = np.where(events[:, 1] == unique)[0] # get rows
        plotData  = np.mean(data[idx,...], 0 )

        for idx, ax in enumerate(axes.flatten()):
            # print(plotData.shape, time.shape)
            ax.plot(time, plotData[..., idx].T, label = unique)
            ax.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter(useMathText = True, useOffset = True) )
            try:
                ax.set_title(cap[idx, 0])
            except:
                pass
        ax.legend(uniques, loc = 'center left' , bbox_to_anchor = (1 , 3.5))
        # plot formatting
        subplots_adjust(hspace = .4)
        mainFrame.set_xlabel('Time[s]', fontsize = 20)
        mainFrame.set_ylabel('mV', fontsize =  20)
        mainFrame.tick_params(labelcolor = 'none',
                              top='off',
                              bottom='off',
                              left='off',
                              right='off')
        mainFrame.tick_params(axis = 'x', pad = 20)
        mainFrame.tick_params(axis = 'y', pad = 20)
    # show()
    return fig


def plotTF(data,
          events,
          cap = None,
          fSample = 100,
          waveletRange = np.linspace(1, 40)):
    '''
    Plots a time frequency representation of the data using complex wavelet convolution
    inputs :
        data        : trial x time x channel
        events      : event type x event value
        cap         : cap file indicating the location of the sensors
        fSample     : sampling rate used for constructing the time vector
        waveletRange: range of the wavelets to plot
    '''
    uniques = np.unique(events[:, 1])
    from matplotlib import ticker
    convData = {}
    # for every condition average the tf decomposition
    for unique in uniques:
        dataIdx = np.where(events[:, 1] == unique)[0] # get the correct indices
        rData   = data[dataIdx,...].flatten()         # reshape 1D array
        cw = scipy.signal.cwt(rData,                  # convolve complex morlet wavelets
                               scipy.signal.ricker,
                               waveletRange)
        cw = cw.reshape(cw.shape[0], *data[dataIdx, ...].shape) # nWavelets x trials x time x channels
        cw = np.mean(cw,1)                                      # average the trials

        # plot
        nCols = 2
        nRows = int(np.ceil(cw.shape[-1] / nCols))
        fig, axes = subplots(nrows = nRows,
                             ncols = nCols,
                             sharex = 'all',
                             sharey = 'all')

        mainFrame = fig.add_subplot(111, frameon = False)
        maxTime = data.shape[1] * 1/fSample
        for idx, ax in enumerate(axes.flatten()):
            im =     ax.imshow(cw[..., idx],
                     aspect = 'auto',
                     interpolation = 'bicubic',
                     extent = [0, maxTime, waveletRange[0], waveletRange[-1]])
            c = colorbar(im, ax = ax)
            c.set_label('Power', fontsize = 15)
            try:
                ax.set_title(cap[idx,0])
            except:
                continue

        subplots_adjust(hspace = .1)
        mainFrame.set_xlabel('Time [s]', fontsize = 20)
        mainFrame.set_ylabel('mV', fontsize = 20)
        [mainFrame.tick_params(axis = i, pad = 30) for i in ['x', 'y'] ]

        fig.suptitle(unique)
        mainFrame.tick_params(labelcolor = 'none',
                              top        = 'off',
                              bottom     = 'off',
                              left       = 'off',
                              right      = 'off')
        mainFrame.set_xlabel('Time [s]')
        mainFrame.set_ylabel('Frequency [Hz]')
        mainFrame.tick_params(axis = 'x', pad = 30)
        mainFrame.tick_params(axis = 'y', pad = 30)

    # show()
    return fig
def plotSpect(data, events):
    '''
    Plots the frequency spectrum per channel using Welch's method
    Inputs:
        data  : trial x time x channel
        events: event type x event value
    '''

    import scipy
    uniques = np.unique(events[:,1])
    fig, axes = subplots(5,2, sharex = 'all', sharey = 'all')
    axx = fig.add_subplot(111, frameon = 0)
    for unique in uniques:
        idx = np.where(events[:,1] == unique)
        tmpData = data[idx,...].squeeze()
        for idx, channelData in enumerate(tmpData.T):
            tmp = 0
            for trialData in channelData.T:
                f, px = scipy.signal.welch(trialData, 250)
                tmp += px
            axes.flatten()[idx].plot(f, tmp / channelData.shape[1])

            axes.flatten()[idx].set_title(cap[idx, 0])
            axes.flatten()[idx].set_xlim([0,30])
            axes.flatten()[idx].yaxis.set_major_formatter(
                matplotlib.ticker.ScalarFormatter(useMathText = True, useOffset = False) )
            xlim([0,40])
    xlabel('Frequency [Hz]', fontsize = 20)
    ylabel('Power', fontsize = 20)
    subplots_adjust(hspace = .4)

    axes.flatten()[idx].legend(uniques, loc = 'center left' , bbox_to_anchor = (1 , 3.5))
    tick_params(          labelcolor = 'none',
                          top        = 'off',
                          bottom     = 'off',
                          left       = 'off',
                          right      = 'off')
    tick_params(axis = 'x', pad = 5)
    tick_params(axis = 'y', pad = 20)
    # show()
    return fig
if __name__ == '__main__':

    from h5py import File
    from classification import SVM
    subjectNumber = 4
    file = '../Data/calibration_subject_{0}.hdf5'.format(subjectNumber)
    with File(file) as f:
        for i in f: print(i)
        procDataIM   = f['procData/IM'].value
        procDataERN  = f['procData/ERN'].value
        eventsIM     = f['events/IM'].value
        eventsERN    = f['events/ERN'].value
        rawDataIM    = f['rawData/ERN'].value
        fSample      = f['fSample'].value
        cap          = f['cap'].value





    # print(procDataIM.shape)
    import preproc

    # plot the ERP of ERN
    # fig = plotERP(procDataERN, eventsERN, cap = cap, fSample = fSample)
    # fig.savefig('../Figures/ERP_subject_{0}.pdf'.format(subjectNumber))
    # # show(fig)
    #
    # fig = plotERP(procDataIM, eventsIM, cap = cap, fSample = fSample)
    #
    # # plot the spectrum of imagined movement
    # fig = plotSpect(procDataIM, eventsIM)
    # fig.savefig('../Figures/Spectrum_subject_{0}.pdf'.format(subjectNumber))

    # fig = plotTF(procDataIM, eventsIM, cap = cap, fSample = fSample)
    # show()
    # # plot the confusion of ERN and imagined movement
    # modelIM, fig = SVM(procDataIM, eventsIM, fft = 1)
    # fig.savefig('../Figures/Confusion_subject_{0}.pdf'.format(subjectNumber))

    # show()


    runTimes  = np.genfromtxt('../Data/BrainRunnerResults.txt', dtype = None)

    playerType = np.unique(runTimes[1:,:][:,2])
    xticker = 0
    fig, ax = subplots()
    for player in playerType:
        if player == '1':
            playerString = 'Human'
        else:
            playerString = 'Bot'
        idx = np.where(runTimes == player)
        times, types, _ = list(runTimes[idx[0], ...].T)
        times = np.array(times, dtype = float)
        modes = np.unique(types)
        for mode in modes:
            jdx         = np.where(mode == types)
            plotRunTime = times[jdx]
            x           = range(len(plotRunTime))
            # print(len(x), xticker)
            stdTime     = np.std(plotRunTime)
            ax.errorbar(x, plotRunTime, stdTime, label = '{0} {1}'.format(playerString, mode))
            # print(type(x), xticker)
            if len(x) > xticker:
                xticker     = len(x)
    ax.legend(loc = 0)
    ax.set_xlabel('Run')
    ax.set_xticks(range(xticker))
    ax.set_ylabel('End Time (s)')
    fig.savefig('../Figures/BrainRunnerTimes.pdf')
    # show()
