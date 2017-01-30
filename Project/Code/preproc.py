from __future__ import division, print_function
import numpy as np
import scipy, scipy.signal
import sklearn
from pylab import *
from h5py import File
def detrend(data, dim = 1, type = "linear"):
    '''Removes trends from the data.

    Applies the scipy.scipy.signal.detrend function to the data, this numpy function
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
    # print(data.shape)
    # assert 0
    # X = data.reshape(-1, data.shape[-1] )
    # print(X.shape)
    # x = scipy.signal.detrend(X, axis = 0)

    return scipy.signal.detrend(data, axis = 1)
def badChannelRemoval(data, x = 3):
    '''
    assumes data is event, time, channels
    returns array of data that should be analyzed
    '''
    std = np.std(data)
    # filter with x standard deviations
    events, time, badChans = np.where(abs(data) >  x * std) # bug fixed, i did the  reverse selection!
    # set all channels to useablewith File(fileCalibration) as f:
                # testData = np.array(testData)
                # f.create_dataset('test', data = testData)

    useable = np.ones(data.shape[-1], dtype = bool)

    # if there is an outlier dont use that channel
    useable[np.unique(badChans)] = False
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

    Notch = np.array([49,51]) / fs
    bb,aa = scipy.signal.butter(N = N, Wn = Notch, btype = 'bandstop')
    b, a = scipy.signal.butter(N = N, Wn = band, btype = filter_type)
    fdata = scipy.signal.filtfilt(bb,aa, data, method ='gust', axis = dim)
    fdata = scipy.signal.filtfilt(b,a, fdata, method = 'gust', axis = dim)
    return fdata


import numpy as np



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

def stdPreproc(data, band,  hdr, cap = None, calibration = 1):
    '''
    Function performs:
            Detrends the data
            Removes bad channels
            Filters the data
    '''
    # linear detrending
    data       = scipy.signal.detrend(data, axis = 1)

    # bad channel removal
    # power      = np.sum(abs(np.fft.fft(data, axis = 1))**2, 0)
    # rData      = data.reshape(-1, data.shape[-1])       # keep channels cat epochs + time
    # power      = abs(np.fft.fft(rData, axis = 0))**2    # total power(?)

    # meanPower  = np.mean(power, axis = 0)               # mean channel power
    # stdPower   = np.std(power,  axis = 0)               # variance channel power
    meanData = np.mean(data)    # global mean
    stdData  = np.std(data)     # global std

    if calibration :
        # remove channels with high power
        #_, remove  = np.where(power > meanPower + 3 * stdPower)
        # print(power, meanPower, remove, power.shape)
        #uniques    = np.unique(remove)

        _,_,badChannels         = np.where(
                                    np.logical_or(
                                    data > meanData + 3 * stdData,
                                    data < meanData - 3 * stdData))
        badChannels             = np.unique(badChannels)
        channels                = np.ones(data.shape[-1], dtype = bool)
        channels[badChannels] = 0
        #channels[uniques] = 0
        #data       = data [..., channels]
    else:
        channels = None

    # feature normalization
    data       = (data - meanData) / stdData


    #temporal filter
    data        = butterFilter(data, band = band, hdr = hdr)

    # spatial filter
    data        = car(data)


    if cap != None:
        print('Removing cdatahannels :\n', cap[(channels-1)*-1, 0])

    return data, channels









if __name__ == '__main__':
    from scipy import signal
    from h5py import File
    import mne
    import sklearn.preprocessing
    with File('../Data/calibration_subject_MOCK_33.hdf5', 'r') as f:
        rawData = f['rawData/IM'].value
        procData = f['procData/IM'].value
        # events  = f['events'].value
        # cap     = f['cap'].value


        procData, _    = stdPreproc(rawData, [0, 14], 250)
        print(procData.shape)

        # binnedData = eventSeparator(procData, events)
        # rickerWavelet(binnedData)
        # ft = abs(np.fft.fft(procData, axis = 1))**2
        #
        # fig, ax = subplots()
        # ax.plot(ft[0, :50 , :])
        # show()
        # binnedData = eventSeparator(procData, events)
        # plotERP(binnedData, cap)
        # rickerWavelet(binnedDat   a)
        # rickerWavelet(binnedData)
        #
        # pos = binnedData['negative']
        # mpos = np.mean(pos,0)
        # mpos = np.mean(mpos[:25], 0)
        # print(mpos.shape)
        # font = {'family' : 'normal',
        # 'weight' : 'bold', 'size'   : 22}
        # head_pos = {'center': cap[:,1], 'scale': cap[:,2]}
        # matplotlib.rc('font', **font)
        # da = array([cap[:,1], cap[:,2]], dtype = int)
        #
        # x = np.sin(da[0,:]) * np.cos(da[1,:])
        # y = np.sin(da[0,:]) * np.sin(da[1,:])
        # pos = np.vstack((x,y))
        # print(pos.shape)
        # mne.viz.plot_topomap(mpos, pos.T, names = cap[:,0], sensors = False, show_names = True, vmin = -1, vmax = 1)
        #
        # layout = mne.channels.read_layout('../../Buffer/resources/caps/cap_tmsi_mobita_im.txt')
        # print(layout)
        # plotERP(binnedData, cap)ricker
        # rickerWavelet(binnedData)


        # plotERP(test, events, )
