from __future__ import division
import numpy as np
import scipy.signal as signal
from pylab import *
from h5py import File
def detrend(data, dim = 1, type = "linear"):
    '''Removes trends from the data.

    Applies the scipy.signal.detrend function to the data, this numpy function
    offers to types of detrending:

    linear - the result of a linear least-squares fit to data is subtracted
    from the data.
    constant - the mean of the data is subtracted from the data.

    Parameters
    ----------
    data : list of datapoints (numpy arrays) or a single numpy array.
    dim : the axis along which detrending needs to be applied
    type : a string that indicates the type of detrending should either
    be "linear" or "constant"

    Returns
    -------
    out : a clone of data on wich the detrending algorithm has been applied

    Examples
    --------
    >>> data, events = f250tc.getData(0,100)
    >>> data = preproc.detrend(data,type="constant")
    >>> data = bufhelp.gatherdata("start",10,"stop")
    >>> data = preproc.detrend(data)
    '''


    X = signal.detrend(data, axis=dim)
    return X

def badChannelRemoval(data, x = 3):
    '''
    assumes data is event, time, channels
    returns array of data that should be analyzed
    '''
    std = np.std(data)
    # filter with x standard deviations
    events, time, badChans = np.where(abs(data) <  x * std)
    # set all channels to useable
    useable = np.ones(data.shape[-1], dtype = bool)

    # if there is an outlier dont use that channel
    useable[np.unique(badChans)] = True
    return useable


def butterFilter(data, band, N = 2, hdr = 100, dim = 1, filter_type = 'lowpass'):
    '''
    Simple butter worth filter
    filter defaults to the first axis (assumes time)
    '''
    if type(hdr) == int:
        fs = hdr
    else:
        fs = hdr.fSample
    band = np.array(band) / float(fs)
    if len(band) == 2:
        filter_type = 'bandpass'

    b, a = signal.butter(N = N, Wn = band, btype = filter_type)
    fdata = signal.filtfilt(b,a, data, method = 'gust', axis = dim)
    return fdata


import numpy as np

def plotERP(data, events, cap,  title = None):

    types       = events[:, 1]
    uTypes      = np.unique(types)
    erp = np.zeros( ( len(uTypes), data.shape[1], data.shape[2] ) )
    label = []
    for idx, type in enumerate(uTypes):
        findIt = np.where(types == type)[0]
        # conditions = np.unique(types[findIt, 1])
        # print(findIt)
        label.append(type)
        meaned =  np.mean(data[findIt, :, :], 0 )
        # print(meaned.shape)
        erp[idx, :, :] = meaned
    if erp.shape[-1] % 2 == 0:
        nRows = nCols =  erp.shape[-1] / 2
    else:
        nRows = (erp.shape[-1] + 1) // 2
    nCols = erp.shape[-1] // nRows

    fig, axs = subplots(nrows = int(nRows),
                        ncols = int(nCols), sharex = 'all', sharey = 'all')
    fig.add_subplot(111, frameon=False)
    for idx, ax in enumerate(axs.flatten()):
        ax.plot(erp[:, :, idx].T)
        ax.set_title(cap[idx,0])


    ax.legend(label, ncol = erp.shape[0],
            loc = 'upper center', bbox_to_anchor = (-.15, -.5)) # centers under all subplots
    ylab = 'mV'
    xlab = 'Time[step]'
    xlabel(xlab, fontsize = 20)
    ylabel(ylab, fontsize = 20)
    tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # fig.text(.05, .5, ylab, rotation = 90, fontsize = 20) # ylabel
    # fig.text(.5 - 1/(float(2*len(xlab))), .06, xlab, fontsize = 20)
    print(1/float(len(xlab)))
    savefig('../Figures/ERP_Lisa.png')
    show()


def car(data):
  '''
  Return a common average reference (CAR) spatial filter for n channels.
  The common average reference is a re-referencing scheme that is
  commonly used when no dedicated reference is given. Since the
  average signal is subtracted from each sensor's signal, it reduces
  signals that are common to all sensors, such as far-away noise.
  Parameters
  ----------
  n : int
    The number of sensors to filer.
  Returns
  -------
  W : array
    Spatial filter matrix of shape (n, n) where n is the number of
    sensors. Each row of W is single spatial filter.
  Examples
  --------
  >>> car(4)
  array([[ 0.75, -0.25, -0.25, -0.25],
         [-0.25,  0.75, -0.25, -0.25],
         [-0.25, -0.25,  0.75, -0.25],
         [-0.25, -0.25, -0.25,  0.75]])
  '''
  n = data.shape[-1]
  W = np.eye(n) - 1 / float(n)
  return  data.dot(W)

def stdPreproc(data, band,  hdr, cap = None):
    '''
    Function performs:
            Detrends the data
            Removes bad channels
            Filters the data
    '''
    global events
    data        = detrend(data)                 # detrend
    # plotERP(data, events, cap)
    data        = signal.detrend(data,          # demean
                                 axis = 2,
                                 type = 'constant')
    data        = car(data)                     # spatial filter
    # plotERP(data, events, cap)
    data        = butterFilter(data,            # filter
                               band = band,
                               hdr = hdr)

    plotERP(data, events, cap)
    useable     = badChannelRemoval(data)       # remove bad channels
    print(useable)



    data        = data[:, :, useable]
    return data
if __name__ == '__main__':
    from h5py import File
    with File('../Data/calibration_subject_4.hdf5') as f:
        rawData = f['rawData'].value
        events  = f['events'].value
        cap     = f['cap'].value

        test = stdPreproc(rawData,[0, 100], 250, cap)
        # plotERP(test, events, )
