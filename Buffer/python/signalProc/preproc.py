from __future__ import division
import numpy as np
import scipy.signal as signal
from pylab import *
from h5py import File
def detrend(data, dim=1, type="linear"):
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
    >>> data, events = ftc.getData(0,100)
    >>> data = preproc.detrend(data,type="constant")
    >>> data = bufhelp.gatherdata("start",10,"stop")
    >>> data = preproc.detrend(data)
    '''

    # if not isinstance(type,str):
    #     raise Exception("type is not a string.")
    #
    # if type!="linear" and type!="constant":
    #     raise Exception("type should either be linear or constant")
    #
    # if not isinstance(data,np.ndarray):
    #     X = concatdata(data)
    # elif isinstance(data,np.ndarray):
    #     X = data
    # else:
    #     raise Exception("data should be a numpy array or list of numpy arrays.")

    X = signal.detrend(data, axis=dim)
    return X

def badChannelRemoval(data, x = 3):
    '''
    assumes data is event, time, channels
    returns array of data that should be analyzed
    '''
    std = np.std(data)
    # filter with x standard deviations

    events, time, bad_chans = np.where(abs(data) >  x * std)
    # set all channels to useable
    useable = np.ones(data.shape[-1], dtype = bool)
    # if there is an outlier dont use that channel
    useable[np.unique(bad_chans)] = False
    return useable


def butter_filter(data, band, hdr = 100, dim = 1, filter_type = 'lowpass'):
    '''
    Simple butter worth filter
    filter defaults to the first axis (assumes time)
    '''

    fs = hdr.fSample
    band = np.array(band)
    print(band / fs)
    # fs = hdr
    if len(band) == 2:
        filter_type = 'bandpass'
    else:
        filter_type = 'filter_type'
    b, a = signal.butter(N = 1, Wn = band / float(fs), btype = filter_type)
    fdata = signal.filtfilt(b,a, data, axis = dim)
    return fdata

def formatData(data, events):
    # reshape data to 2D for classifciation
    nSamples = data.shape[1]
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[-1])
    eventClass = np.zeros(data.shape[0])

    # obtain unique labels in the events [target / stimuli]
    uniques = np.unique([x.type for x  in events])
    # map them to dict to construct numeric class labels
    classLabels = {}
    for idx, label in enumerate(uniques):
        classLabels[label] = idx

    #  convert them to list according to the stacked 2D data
    eventClassLabels = np.zeros(data.shape[0])
    c = 0
    for event in events:
        classLabel = classLabels[event.type]
        for i in range(nSamples):
            eventClassLabels[c] = classLabel
            c += 1

    return data, eventClassLabels

if __name__ == '__main__':
    with File('test_script.hdf5') as f:
        data = f['data'].value

    def plotter(data):
        fig, ax = subplots(data.shape[-1])
        for i, axi in enumerate(ax):
            axi.plot(data[1,:,i])

    data = detrend(data)
    useable = badChannelRemoval(data)
    data = data[:,:, useable]
    data = butter_filter(data, [0, 30 ])
    print(useable)

    show(1)
