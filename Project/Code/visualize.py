from __future__ import print_function, division

from pylab import *
import numpy as np
import scipy, scipy.signal


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
            try:
                ax.plot(time, plotData[..., idx].T, label = unique)
                ax.legend()
                ax.set_title(cap[idx, 0])
            except:
                pass

        # plot formatting
        subplots_adjust(hspace = .4)
        mainFrame.set_xlabel('Time[s]', fontsize = 20)
        mainFrame.set_ylabel('mV', fontsize =  20)
        mainFrame.tick_params(labelcolor = 'none',
                              top='off',
                              bottom='off',
                              left='off',
                              right='off')
        mainFrame.tick_params(axis = 'x', pad = 30)
        mainFrame.tick_params(axis = 'y', pad = 30)
    show()


def plotTF(data, events, cap = None, fSample = 100):
    uniques = np.unique(events[:, 1])
    from matplotlib import ticker
    wavelets = np.logspace(0, 1.2, 25)
    convData = {}
    # for every condition average the tf decomposition
    for unique in uniques:
        dataIdx = np.where(events[:, 1] == unique)[0] # get the correct indices
        rData   = data[dataIdx,...].flatten()         # reshape 1D array
        cw = scipy.signal.cwt(rData,                  # convolve complex morlet wavelets
                               scipy.signal.ricker,
                               wavelets)
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
                     origin = 'lower',
                     interpolation = 'bicubic',
                     extent = [0, maxTime, wavelets[0], wavelets[-1]])
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

    show()


if __name__ == '__main__':
    from h5py import File
    with File('../Data/calibration_subject_MOCK_33.hdf5') as f:
        for i in f: print(i)
        procDataIM = f['procData/IM'].value
        eventsIM   = f['events/IM'].value

    print(procDataIM.shape)

    # plotERP(procDataIM, eventsIM, cap = None)
    plotTF(procDataIM, eventsIM)
