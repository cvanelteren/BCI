from    __future__          import print_function
from    h5py                import File, special_dtype
from    pylab               import *
from    systemHelper        import checkOverwrite


import warnings
# with warnings.catch_warnings():
# warnings.filterwarnings('ignore')

import  classification, preproc, bufhelp, os
import  numpy as np
import time

from scipy.signal import detrend
ftc, hdr = bufhelp.connect() # connect to buffer

# get cap file
filePath    = '../../Buffer/resources/caps/cap_tmsi_mobita_im.txt'
capFile     = np.loadtxt(filePath, dtype=str)
nChans      = len(capFile)  # mobita outputs 37, redundant channels remove them
#nChans      = 4


# SET THE SUBJECT NUMBER
dataDir        = '../Data/'                 # storage of directory
conditionType  = 'calibration_subject_'     # calibration file
subjectNumber  =  100                         # subject number

# storage file
fileCalibration = dataDir + conditionType + str(subjectNumber) + '.hdf5'
# BUFFER PARAMETERS
trlen_ms       = 600

# event.type, event.value to watch for
calibrationCatchEvents = [\
                        'target', 'feet',\
                        'target', 'right hand',\
                        'target', 'left hand',\
                        'feedback','20',\
                        'feedback','negative'\
                        ]


# fileCalibration = checkOverwrite(dataDir, conditionType, subjectNumber)

print("Waiting for start event.")
run            = True
while run:
    e = bufhelp.waitforevent('start', 1000, 0)

    if e is not None:
        # print('Found event')calibration
        if e.value == "calibration":
            print("Calibration phase")


            # catch events
            data, events, stopevents = bufhelp.gatherdata(\
                                        calibrationCatchEvents, \
                                        trlen_ms, ("calibration", "end"), \
                                        milliseconds=True, verbose = False)

            # convert to arrays and save to disk
            data = np.array(data) # cut off the redundant channels
            print(data.shape)
            data = data[:, :, :nChans]
            dt = special_dtype(vlen=bytes)
            ev = np.array([(event.type, event.value) for event in events])
            # specify [lowerBound, upperBound] bandpass filter range
            filterBand = [0, 40]
            procData = preproc.stdPreproc(data, filterBand, hdr)
            with File(fileCalibration, 'w') as f:
                # f.create_dataset('targets', data = tmp)
                f.create_dataset('rawData',       data=data)
                f.create_dataset('events',        data=ev, dtype=dt)
                f.create_dataset('processedData', data=procData)
                f.create_dataset('cap',           data=capFile)
            print("End calibration phase")

        # load data from disk; train classifier
        elif e.value == "train":
            print("Training classifier")
            with File(fileCalibration, 'r') as f:
                ev       = f['events'].value
                procData = f['processedData'].value
            print(procData.shape)
            modelMovement = classification.SVM(procData, ev, type = 'target',string='im')[0]
            modelERN      = classification.SVM(procData, ev, type = 'feedback',string='ern')[0]

            bufhelp.sendEvent("training", "done")

        # interface with the game
        elif e.value == "test":
            print("Feedback phase")
            # nChans = hdr.nChannels
            dt =  1 / hdr.fSample

            # PARAMETERS
            plotTime = 3
            nPoints  = int((trlen_ms/1e3) / dt)
            print(hdr.fSample)
            # timeSec = linspace(-nPoints, 0, nPoints)
            # print(nPoints)

            keep = True
            while keep:
                bufferStorage = zeros((nPoints, nChans))
                # print(bufferStorage.shape, nChans)
                # get latest samples and plot them
                tic = t  =  time.time()
                predsIM = []
                predsERN = []
                i = 0
                while abs(tic - time.time()) < plotTime:
                    idx = ftc.getHeader().nSamples  - 1
                    lastSample =  ftc.getData((idx,idx))
                    bufferStorage            = roll(bufferStorage, -1, 0)
                    # print(lastSample.shape)
                    bufferStorage[-1, :]     = lastSample[0, :nChans]
                    # print(bufferStorage[-2,:])
                    pause(.01)
                    # i+= 1
                    if  t > trlen_ms / 1e3:
                        bufferStorage = detrend(bufferStorage, 0, type = 'linear')
                        bufferStorage = detrend(bufferStorage, 0, type = 'constant')
                        # bufferStorage = np.array(bufferStorage.flatten(), ndmin = 2 )
                        pred = modelMovement.predict_proba(bufferStorage.reshape(bufferStorage.shape[0] * bufferStorage.shape[1])) # prediction
                        predsIM.append([pred])
                        pred = modelERN.predict_proba(bufferStorage.reshape(bufferStorage.shape[0] * bufferStorage.shape[1])) # prediction
                        predsERN.append([pred])
                        # bufferStorage = zeros((nPoints, nChans)) # flush
                        i += 1
                    t = time.time() - tic

                predsIM = np.array(predsIM).squeeze()
                print(predsIM)
                predsERN = np.array(predsERN).squeeze()
                weightingIM = np.arange(start=predsIM.shape[0]+1,stop=1,step=-1)[:,None]
                weightingERN = np.arange(start=predsERN.shape[0]+1,stop=1,step=-1)[:,None]

                predsIM /= weightingIM
                predsERN /= weightingERN

                maxPredIM = np.max(predsIM, axis = 0)
                maxPredERN = np.max(predsERN, axis = 0)
                # print('>', maxPredIM)
                bufhelp.sendEvent('clsfr.prediction.im', maxPredIM)
                bufhelp.sendEvent('clsfr.prediction.ern', maxPredERN)
                #bufhelp.sendEvent('clsfr.prediction.im', pred)

        elif e.value == "exit":
            run = False
        # print(e.value)
        print("Waiting for start event.")
