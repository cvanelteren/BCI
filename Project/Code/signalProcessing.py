from    __future__          import print_function
from    h5py                import File, special_dtype
from    pylab               import *
from    systemHelper        import checkOverwrite

import  classification, preproc, bufhelp, os
import  numpy as np

ftc, hdr = bufhelp.connect() # connect to buffer

# get cap file
filePath    = '../../Buffer/resources/caps/cap_tmsi_mobita_im.txt'
capFile     = np.loadtxt(filePath, dtype=str)
nChans      = len(capFile)  # mobita outputs 37, redundant channels remove them


# SET THE SUBJECT NUMBER
dataDir        = '../Data/'                 # storage of directory
conditionType  = 'calibration_subject_'     # calibration file
subjectNumber  =  0                         # subject number

# storage file
fileCalibration = checkOverwrite(dataDir, conditionType, subjectNumber)

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
            data = np.array(data[:, :, nChans]) # cut off the redundant channels
            dt = special_dtype(vlen=bytes)
            ev = np.array([(event.type, event.value) for event in events])
            # specify [lowerBound, upperBound] bandpass filter range
            filterBand = [0, 60]
            processedData = preproc.stdPreproc(data, filterBand, hdr)
            with File(fileCalibration, 'w') as f:
                # f.create_dataset('targets', data = tmp)
                f.create_dataset('rawData',       data=data)
                f.create_dataset('events',        data=ev, dtype=dt)
                f.create_dataset('processedData', data=processedData)
                f.create_dataset('cap',           data=capFile)
            print("End calibration phase")

        # load data from disk; train classifier
        elif e.value == "train":
            print("Training classifier")
            bufhelp.sendEvent("training", "done")

        # interface with the game
        elif e.value == "test":
            print("Feedback phase")
            keep = True
            while keep:
                data, events, stopevents = bufhelp.gatherdata(["stimulus"],\
                                                            trlen_ms,[("run","end"),\
                                                            ('test', 'end')], \
                                                            milliseconds=True,\
                                                             verbose = False)

                if stopevents.type == 'test' and stopevents.value == 'end':
                    keep = False
                else:
                    # specify [lowerBound, upperBound] bandpass filter range
                    filterBand = [0, 60]
                    # linear detrend, filter, average across epochs
                    processedData = preproc.stdPreproc(data, filterBand, hdr)
                    pred = FeedbackSignals.genPrediction(processedData, model,
                                                         events)
                    # sanity check
                    print('prediction', pred)
                    bufhelp.sendEvent("classifier.prediction", pred)
                    data = np.array(data)
                    ev = np.array(
                        [(event.type, event.value) for event in events])
                    # save the test phase
                    with File(fileTest, 'w') as f:
                        # f.create_dataset('targets', data = tmp)
                        f.create_dataset('rawData', data=data)
                        f.create_dataset('events', data=ev, dtype=dt)
                        f.create_dataset('processedData', data=processedData)
                        # f.create_dataset()

        elif e.value == "exit":
            run = False
        # print(e.value)
        print("Waiting for start event.")
