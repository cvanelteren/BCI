from    __future__          import print_function
from    h5py                import File, special_dtype
from    pylab               import *
from    systemHelper        import checkOverwrite
from    scipy               import signal
import sklearn
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
subjectNumber  = 1                           # subject number

if hdr.fSample == 100:                          # debug case
    nChans          = 4
    conditionType   = conditionType + 'MOCK_'
    # hdf = 250
# storage file
fileCalibration = dataDir + conditionType + str(subjectNumber) + '.hdf5'
# BUFFER PARAMETERS
trlen_ms       = 1000

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
        # print('dataFound event')calibration
        if e.value == "calibration":
            print("Calibration phase")

            #fileCalibration = checkOverwrite(dataDir, conditionType, subjectNumber)
            # catch events
            data, events, stopevents = bufhelp.gatherdata(\
                                        calibrationCatchEvents, \
                                        trlen_ms, ("calibration", "end"), \
                                        milliseconds=True, verbose = False)

            print('yo')
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
            # nPointsTrial  = int((trlen_ms/1e3) / dt)
            nSamples      = int(3 / (trlen_ms / 1e3))
            nPoints       =  int(3 / dt)


            weight = np.exp(- np.linspace(3,0,nSamples))[:, None].T

            print('Sampling rate', hdr.fSample)
            # timeSec = linspace(-nPoints, 0, nPoints)
            # print(nPoints)

            keep = True
            bufferStorage = zeros((nPoints, nChans))
            testData = []

            while keep:
                endSample, startEvent    = ftc.poll()                                 # get current index sample
                startSample              = endSample - nPoints + 1                    # compute end sample
                # _, stopEvent             = ftc.wait(nPoints, -1, timeout = nPoints)   # 3 sec timout
                _, stopEvent             = ftc.wait(endSample + nPoints, -1, timeout = 3000)
                events                   = ftc.getEvents((startEvent, stopEvent - 1)) # get events within frame
                for ev in events:                                                     # check for stopping
                    if ev.type == 'test' and ev.value == 'end':
                        keep = False

                # the try command is here because when debugging the event viewer freezes
                # yielding NoneType for data, which will crash; this is a workaround
                try:
                    bufferStorage    = ftc.getData((startSample, endSample))    # grab from buffer
                    bufferStorage    = bufferStorage[:, :nChans]
                    bufferStorage    = bufferStorage.reshape(nSamples, -1)      # reshape nSamples x (time x channels)
                    bufferStorage    = preproc.stdPreproc(bufferStorage, [0,40], hdr)
                    IM               = modelIM.predict_proba(abs(np.fft.fft(bufferStorage, axis = 1))**2)     # compute probability for IM
                    weightedIM       = weight.T  * IM                           # weigh IM
                    maxIMIdx         = np.unravel_index(np.argmax(weightedIM), weightedIM.shape)[0] # compute the max index
                    bufhelp.sendEvent('clsfr.prediction.im',  IM[maxIMIdx, :])


                    endSample, _  = ftc.poll()
                    startSample  =  endSample - 250 + 1
                    ftc.wait(endSample + 250 - 1, -1 , timeout = 1000000)
                    bufferStorage    = ftc.getData((startSample, endSample))
                    bufferStorage    = bufferStorage[:, :nChans]
                    #print(bufferStorage.shape)
                    bufferStorage    = bufferStorage.reshape(1, -1)      # reshape nSamples x (time x channels)
                    bufferStorage    = preproc.stdPreproc(bufferStorage, [0,40], hdr)
                    #print('> ', startSample)
                    ERN              = modelERN.predict_proba(bufferStorage)    # compute probability for ERN
                    #print(ERN)
                    # weightedERN      = weight.T  * ERN                          # weigh ERN
                    # maxERNIdx        = np.unravel_index(np.argmax(weightedERN), weightedIM.shape)[0]

                    # send the events!
                    bufhelp.sendEvent('clsfr.prediction.ern', ERN[0,:])
                except:
                    pass

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
