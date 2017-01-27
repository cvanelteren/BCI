from    __future__          import print_function, division
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
subjectNumber  = 25                           # subject number

if hdr.fSample == 100:                          # debug case
    nChans          = 4
    conditionType   = conditionType + 'MOCK_'
    capFile         = None
    # hdf = 250
# storage file
fileCalibration     = dataDir + conditionType + str(subjectNumber) + '.hdf5'
# BUFFER PARAMETERS
trialLenIM          = 1000 # msec
trialLenERN         = 600  # msec


 # trial length per condition
trialLengthmapping = {'feedback': trialLenERN,
                     'target': trialLenIM}

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
# nChans = hdr.nChannels
dt =  1 / hdr.fSample


nPointsERN      =  int(trialLenERN /(1e3) * hdr.fSample) # number of points per epoch IM
nPointsIM       =  int(trialLenIM / (1e3) * hdr.fSample) # number of points per epoch ERN

print("Waiting for start event.")
run            = True
while run:
    e = bufhelp.waitforevent('start', 1000, 0)

    if e is not None:
        # print('dataFound event')calibration
        if e.value == "calibration":
            print("Calibration phase")

            fileCalibration = checkOverwrite(dataDir, conditionType, subjectNumber)
            # catch events
            data, events, stopevents = bufhelp.gatherdata(\
                                        calibrationCatchEvents, \
                                        trialLengthmapping, ("calibration", "end"), \
                                        milliseconds=True, verbose = False)
            ev                       = np.array([(event.type, event.value) for event in events])

            # separate the trial types
            dataIM    = []
            dataERN   = []
            eventsIM  = []
            eventsERN = [] #  init store lists
            for i, j in zip(data, ev):
                print(i.shape, j.shape)
                if i.shape[0] == nPointsIM:
                    dataIM.append(i)
                    eventsIM.append(j)
                else:
                    dataERN.append(i)
                    eventsERN.append(j)

            eventsIM    = np.array(eventsIM)
            eventsERN   = np.array(eventsERN)

            dataIM      = np.array(dataIM)
            dataERN     = np.array(dataERN)

            print(dataIM.shape, dataERN.shape)
            dataIM      = dataIM[..., :nChans]
            dataERN     = dataERN[...,:nChans]
            # data                    = np.array(data)
            eventDataType           = special_dtype(vlen=bytes)                                 # for storage in hdf5
             # np array of events


            print('IM data shape', dataIM.shape)
            print('ERN data shape', dataERN.shape)
            filterBand = [0, 40]                                                     # filter range
            procDataIM, chanSelectIM   = preproc.stdPreproc(dataIM, filterBand, hdr,  cap  = capFile)
            procDataERN, chanSelectERN = preproc.stdPreproc(dataERN, filterBand, hdr, cap = capFile)
            with File(fileCalibration, 'w') as f:
                # store IM condition
                f.create_dataset('rawData/IM',       data = dataIM)
                f.create_dataset('procData/IM',      data = procDataIM)
                f.create_dataset('events/IM',        data = eventsIM,  dtype = eventDataType)
                f.create_dataset('chanSelector/IM',     data = chanSelectIM)

                # store ERN condition
                f.create_dataset('rawData/ERN',      data = dataERN)
                f.create_dataset('procData/ERN',     data = procDataERN)
                f.create_dataset('events/ERN',       data = eventsERN, dtype = eventDataType)
                f.create_dataset('chanSelector/ERN', data = chanSelectERN)

                if capFile != None:
                    f.create_dataset('cap',          data = capFile)
            print("End calibration phase")

        # load data from disk; train classifier
        elif e.value == "train":
            print('Loading from ' + fileCalibration)
            print("Training classifier")
            with File(fileCalibration, 'r') as f:
                for i in f: print(i)          # file content : debug info
                #events
                eventsIM      = f['events/IM'].value
                eventsERN     = f['events/ERN'].value

                # pre-processed data
                procDataIM   = f['procData/IM'].value
                procDataERN  = f['procData/ERN'].value

                # boolean for removed channels
                chanSelectIM = f['chanSelector/IM'].value
                chanSelectERN = f['chanSelector/ERN'].value
            modelIM  = classification.SVM(procDataIM, eventsIM)
            modelERN = classification.SVM(procDataERN, eventsERN)

            bufhelp.sendEvent("training", "done")

        # interface with the game
        elif e.value == "test":
            print("Feedback phase")
            # PARAMETERS
            plotTime = 3
            # nPointsTrial  = int((trialLenIM/1e3) / dt)


            nTimePointsIM       =  int(1 / dt)
            nSamplesIM          =  int(nTimePointsIM * plotTime *  dt)

            nPointsIM           = nSamplesIM * nTimePointsIM
            print(nPointsIM, nSamplesIM)
            weight = np.exp(- np.linspace(nSamplesIM, 0, nTimePointsIM))[:, None].T

            print('Sampling rate', hdr.fSample)
            # timeSec = linspace(-nPointsIM, 0, nPointsIM)
            # print(nPointsIM)

            keep = True
            # bufferStorage = zeros((nPointsIM, len(chanSelectIM)))
            testData = []

            while keep:
                endSample, startEvent    = ftc.poll()                                 # get current index sample
                startSample              = endSample - nPointsIM + 1                    # compute end sample
                # _, stopEvent             = ftc.wait(nPointsIM, -1, timeout = nPointsIM)   # 3 sec timout
                _, stopEvent             = ftc.wait(endSample + nPointsIM, -1, timeout = 3000)
                events                   = ftc.getEvents((startEvent, stopEvent - 1)) # get events within frame
                for ev in events:                                                     # check for stopping
                    if ev.type == 'test' and ev.value == 'end':
                        keep = False

                # the try command is here because when debugging the event viewer freezes
                # yielding NoneType for data, which will crash; this is a workaround
                # try:
                bufferStorage    = ftc.getData((startSample, endSample))    # grab from buffer
                bufferStorage    = bufferStorage[:, :nChans]
                bufferStorage    = bufferStorage[:, chanSelectIM]
                print(bufferStorage.shape)
                bufferStorage =  bufferStorage.reshape(nSamplesIM, nTimePointsIM, bufferStorage.shape[-1])
                print(bufferStorage.shape)
                bufferStorage, _     = preproc.stdPreproc(bufferStorage, [0,40], hdr)
                print(bufferStorage.shape)
                bufferStorage    = bufferStorage.reshape(nPointsIM, -1).T      # reshape nPointsIM x (time x channels)
                print('after', bufferStorage.shape)
                print(bufferStorage.shape)
                IM               = modelIM.predict_proba(abs(np.fft.fft(bufferStorage, axis = 1))**2)     # compute probability for IM
                print(IM)
                weightedIM       = weight.T  * IM
                assert 0                          # weigh IM
                maxIMIdx         = np.unravel_index(np.argmax(weightedIM), weightedIM.shape)[0] # compute the max index
                bufhelp.sendEvent('clsfr.prediction.im',  IM[maxIMIdx, :])


                endSample, _  = ftc.poll()
                startSample  =  endSample - nPointsERN + 1
                ftc.wait(endSample + nPointsERN - 1, -1 , timeout = 1000000)
                bufferStorage    = ftc.getData((startSample, endSample))
                bufferStorage    = bufferStorage[:, :nChans]
                bufferStorage    = bufferStorage[:, chanSelectERN]
                print(bufferStorage.shape)
                #print(bufferStorage.shape)
                bufferStorage    = bufferStorage.reshape(1, -1).T      # reshape nPointsIM x (time x channels)
                bufferStorage, _    = preproc.stdPreproc(bufferStorage, [0,40], hdr)
                #print('> ', startSample)
                ERN              = modelERN.predict_proba(bufferStorage)    # compute probability for ERN
                #print(ERN)
                # weightedERN      = weight.T  * ERN                          # weigh ERN
                # maxERNIdx        = np.unravel_index(np.argmax(weightedERN), weightedIM.shape)[0]

                # send the events!
                bufhelp.sendEvent('clsfr.prediction.ern', ERN[0,:])
                # except:
                #     pass

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
