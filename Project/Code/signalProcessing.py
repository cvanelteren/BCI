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
subjectNumber  =  4                         # subject number

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

            restCondition = np.where(ev == 'rest')[1]
            useIdx  = len(restCondition) / 3
            np.random.shuffle(restCondition)
            restCondition = restCondition[useIdx:]
            useThese = np.zeros((procData.shape[0]))
            useThese[restCondition] = 1
            number = classification.stupidFct()
            print(number)
            modelMovement, rD = classification.SVM(\
            procData[useThese == 0, :], ev[useThese==0,:], type = 'target',string='im')[0:2]
            modelERN      = classification.SVM(procData, ev, type = 'feedback',string='ern')[0]
            print(modelMovement)

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
            i = 0
            j = 0
            k = 0
            bufferStorage = zeros((nPoints, nChans))
            while keep:
                # print(bufferStorage.shape, nChans)
                # get latest samples and plot them
                tic = t  =  time.time()
                predsIM = []
                predsERN = []
                # dd = []
                idx = ftc.getHeader().nSamples- 1
                lastSample1 = ftc.getData((idx, idx))
                while abs(tic - time.time()) < plotTime:
                    j += 1
                    pause(0.01)
                    idx = ftc.getHeader().nSamples  - 1
                    lastSample2 =  ftc.getData((idx,idx))
                    if not all(lastSample1 == lastSample2):
                        k += 1
                        lastSample1 = lastSample2

                        bufferStorage            = np.roll(bufferStorage, -1, 0)
                        bufferStorage[-1, :]     = lastSample1[0, :nChans]
                        #print('-')
                        #print(bufferStorage)
                        # print(bufferStorage[-2,:])
                        pause(0.01)
                        # i+= 1
                        if  t > trlen_ms / 1e3:
                            tmp = detrend(bufferStorage, 0, type = 'linear')
                            tmp = detrend(tmp, 0, type = 'constant')
                            #tmp  = bufferStorage
                            # bufferStorage = np.array(bufferStorage.flatten(), ndmin = 2 )
                            tmp = tmp.reshape(bufferStorage.shape[0] * bufferStorage.shape[1])[None,:]
                            print(tmp[0])
                            print(tmp.shape)
                            pred = modelMovement.predict_proba(tmp) # prediction

                            predsIM.append([pred])
                            pred = modelERN.predict_proba(tmp) # prediction
                            predsERN.append([pred])
                            # bufferStorage = zeros((nPoints, nChans)) # flush
                            i += 1
                        #dd.append(bufferStorage)
                        print('====')
                        #print(bufferStorage)
                        #if i > 3:
                        #    assert 0
                        t = time.time() - tic
                        # print(predsIM)
                        if (len(predsIM) > 10):
                            predsIM = np.array(predsIM).squeeze()
                            print(predsIM.shape)
                            # print(predsIM[0,0])
                            predsERN = np.array(predsERN).squeeze()
                            weightingIM = np.arange(start=predsIM.shape[0]+1,stop=1,step=-1)[:,None]
                            weightingERN = np.arange(start=predsERN.shape[0]+1,stop=1,step=-1)[:,None]

                            # predsIM     *= weightingIM
                            # predsERN    *= weightingERN
                            print(predsIM)
                            maxPredIM = 0
                            maxPredERN = 0
                            #maxPredIM = np.max(predsIM, axis = 0)
                            #maxPredERN = np.max(predsERN, axis = 0)
                            # print('>', maxPredIM)
                            bufhelp.sendEvent('clsfr.prediction.im', maxPredIM)
                            bufhelp.sendEvent('clsfr.prediction.ern', maxPredERN)
                            #bufhelp.sendEvent('clsfr.prediction.im', pred)
                            predsIM = []
                            predsERN = []
                    # if different sample
                # while 3 seconds loop
        elif e.value == "exit":
            run = False
        # print(e.value)
        print("Waiting for start event.")
