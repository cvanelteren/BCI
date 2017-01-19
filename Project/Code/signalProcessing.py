from    __future__          import print_function
from    h5py                import File, special_dtype
from    pylab               import *
from    systemHelper        import checkOverwrite


import  classification, preproc, bufhelp, os
import  numpy as np
import time
ftc, hdr = bufhelp.connect() # connect to buffer

# get cap file
filePath    = '../../Buffer/resources/caps/cap_tmsi_mobita_im.txt'
capFile     = np.loadtxt(filePath, dtype=str)
nChans      = len(capFile)  # mobita outputs 37, redundant channels remove them
nChans      = 4


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
            data = np.array(data) # cut off the redundant channels
            print(data.shape)
            data = data[:, :, :nChans]
            dt = special_dtype(vlen=bytes)
            ev = np.array([(event.type, event.value) for event in events])
            # specify [lowerBound, upperBound] bandpass filter range
            filterBand = [0, 60]
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
            modelMovement = classification.SVM(procData, ev, type = 'target')[0]
            modelERN      = classification.SVM(procData, ev, type = 'feedback')[0]

            bufhelp.sendEvent("training", "done")

        # interface with the game
        elif e.value == "test":
            print("Feedback phase")
            nChans = hdr.nChannels
            dt =  1 / hdr.fSample

            # PARAMETERS
            plotTime = 3
            nPoints  = int((trlen_ms/1e3) / dt)
            # timeSec = linspace(-nPoints, 0, nPoints)
            print(nPoints)
            weighting = np.array(range(int(3/dt),-1,1))
            keep = True
            while keep:
                bufferStorage = zeros((nPoints, nChans))
                # get latest samples and plot them
                tic = t = tt =   time.time()
                preds = []
                while t < plotTime:
                    idx = ftc.getHeader().nSamples  - 1
                    lastSample =  ftc.getData((idx,idx))
                    bufferStorage            = roll(bufferStorage, 0)
                    bufferStorage[-1, :]     = lastSample
                    print(t)
                    if  t > trlen_ms / 1e3:
                        pred = modelMovement.predict_proba(bufferStorage) # prediction
                        preds.append(pred)
                        bufferStorage = zeros((nPoints, nChans)) # flush
                    t = time.time() - tic
                    print(t)
                preds = np.array(preds)
                preds /= weighting
                print(preds)
                preds = np.argmax(preds, axis = 0)
                bufhelp.sendEvent('clsf.prediction.im', preds)
                print(preds)







                bufhelp.sendEvent('clsfr.prediction.im', pred)
                print(pred)

        elif e.value == "exit":
            run = False
        # print(e.value)
        print("Waiting for start event.")
