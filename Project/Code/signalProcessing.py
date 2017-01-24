from    __future__          import print_function
from    h5py                import File, special_dtype
from    pylab               import *
from    systemHelper        import checkOverwrite
from    scipy               import signal

import warnings
import classification, preproc, bufhelp, os
import numpy as np
import time

from scipy.signal import detrend
ftc, hdr = bufhelp.connect() # connect to buffer

# get cap file
filePath    = '../../Buffer/resources/caps/cap_tmsi_mobita_im.txt'
capFile     = np.loadtxt(filePath, dtype=str)
nChans      = len(capFile)  # mobita outputs 37, redundant channels remove them



# SET THE SUBJECT NUMBER
dataDir        = '../Data/'                     # storage of directory
conditionType  = 'calibration_subject_'         # calibration file
subjectNumber  = 7                              # subject number

if hdr.fSample == 100:                          # debug case
    nChans          = 4
    conditionType   = conditionType + 'MOCK_'

# storage file
fileCalibration = dataDir + conditionType + str(subjectNumber) + '.hdf5'
# BUFFER PARAMETERS
trlen_ms       = 600

# event.type, event.value to watch for
calibrationCatchEvents = [\
                        'target', 'feet',\
                        'target', 'right hand',\
                        'target', 'left hand',\
                        'feedback','positive',\
                        'feedback','negative'\
                        ]

wn = np.array([0, 40]) / hdr.fSample
b, a = signal.butter(2, wn, btype = 'bandpass')


print("Waiting for start event.")
run            = True
while run:
    e = bufhelp.waitforevent('start', 1000, 0)

    if e is not None:
        # print('Found event')calibration
        if e.value == "calibration":
            print("Calibration phase")

            fileCalibration = checkOverwrite(dataDir, conditionType, subjectNumber)
            # catch events
            data, events, stopevents = bufhelp.gatherdata(\
                                        calibrationCatchEvents, \
                                        trlen_ms, ("calibration", "end"), \
                                        milliseconds=True, verbose = False)

            # convert to arrays and save to disk
            data         = np.array(data) # cut off the redundant channels
            data         = data[:, :, :nChans]
            dt           = special_dtype(vlen=bytes)
            ev           = np.array([(event.type, event.value) for event in events])
            # uniqueEvents = np.unique(ev[:,0])

            # specify [lowerBound, upperBound] bandpass filter range
            print('Data shape ', data.shape)
            filterBand = [0, 40]
            procData = preproc.stdPreproc(data, filterBand, hdr)
            with File(fileCalibration, 'w') as f:
                f.create_dataset('rawData',       data = data)
                f.create_dataset('events',        data = ev, dtype = dt)
                f.create_dataset('processedData', data = procData)
                f.create_dataset('cap',           data = capFile)
                # f.create_dataset('mapping',       data = mapping)
            print("End calibration phase")

        # load data from disk; train classifier
        elif e.value == "train":
            print('Loading from ' + fileCalibration)
            print("Training classifier")
            with File(fileCalibration, 'r') as f:
                for i in f: print(i)          # file content : debug info
                events       = f['events'].value
                procData     = f['rawData'].value
                procData = preproc.stdPreproc(procData, [0,40], hdr)
                # mapping  = f['mapping'].value

            modelIM, modelERN = classification.SVM(procData, events)
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
            testData = []

            while keep:
                # print(bufferStorage.shape, nChans)
                # get latest samples and plot them
                tic = t  =  time.time()
                predsIM = []
                predsERN = []
                # dd = []
                idx = ftc.getHeader().nSamples- 1
                lastSample1 = ftc.getData((idx, idx))

                # if exit event is given exit
                event       = ftc.getEvents()[-4:] # hacky way
                for e in event:
                    if e.type == 'test' and e.value == 'end':
                        keep = False
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
                        # i+= 1
                        if  t > trlen_ms / 1e3:
                            tmp = detrend(bufferStorage, 0, type = 'linear')
                            tmp = detrend(tmp, 0, type = 'constant')
                            tmp = signal.filtfilt(b,a, tmp, method = 'gust', axis = 0)
                            # store test data
                            testData.append(tmp)
                            #tmp  = bufferStorage
                            # bufferStorage = np.array(bufferStorage.flatten(), ndmin = 2 )
                            tmp = tmp.reshape(bufferStorage.shape[0] * bufferStorage.shape[1])[None,:]
                            #print(tmp[-1])
                            #print(tmp.shape)
                            pred = modelIM.predict_proba(tmp) # prediction

                            predsIM.append([pred])
                            pred = modelERN.predict_proba(tmp) # prediction
                            predsERN.append([pred])
                            i += 1

                        t = time.time() - tic
                        # print(predsIM)
                        if (len(predsIM) > 10):
                            predsIM = np.array(predsIM).squeeze()
                            #print(predsIM.shape)
                            # print(predsIM[0,0])
                            predsERN        = np.array(predsERN).squeeze()
                            weightingIM     = np.arange(start=predsIM.shape[0]+1,stop=1,step=-1)[:,None]
                            weightingERN    = np.arange(start=predsERN.shape[0]+1,stop=1,step=-1)[:,None]
                            maxPredIM       = np.min(predsIM, axis = 0)
                            maxPredERN      = np.min(predsERN, axis = 0)
                            # print('>', maxPredIM)
                            bufhelp.sendEvent('clsfr.prediction.im', 1 - maxPredIM)
                            bufhelp.sendEvent('clsfr.prediction.ern', 1 - maxPredERN)
                            #bufhelp.sendEvent('clsfr.prediction.im', pred)
                            predsIM = []
                            predsERN = []
            print('Ending test phase\n storing data...')
            # with File(fileCalibration) as f:
            #     testData = np.array(testData)
            #     f.create_dataset('test', data = testData)

                        # if different sample
                    # while 3 seconds loop
        elif e.value == "exit":
            run = False
        # print(e.value)
        print("Waiting for start event.")
