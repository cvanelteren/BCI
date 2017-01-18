#!/usr/bin/python
#rewritten script in python from the matlab adaptation
#  Author:  Casper van Elteren
from __future__ import print_function
import classification, preproc, time, FeedbackSignals
import numpy as np
from h5py import File, special_dtype
from pylab import *
# from bufferInterface import bufhelp
import bufhelp

# connect to buffer
ftc, hdr = bufhelp.connect()

# storage for the data
dataDir = '../Data/'
file = dataDir + 'test_subject.hdf5'
print('Storing data in: \n\t ', file)

# p300 occurs up to 500 ms after stimulus
# taking 100 ms more than expectation
trlen_ms = 600
run = True

print("Waiting for start event.")
while run:
    e = bufhelp.waitforevent('start',1000, 0)

    # print(':',type(e))
    # print(e)
    if e is not None:
        # print(e[0], e[1])
        # print('Found event')calibration
        if e.value == "calibration":
            print("Calibration phase")
            # catch events
            data, events, stopevents = bufhelp.gatherdata(\
                                        ['target','stimulus'], \
                                        trlen_ms, ("calibration", "end"), \
                                        milliseconds=True, verbose = False)

            # convert to arrays and save to disk
            data            = np.array(data)
            dt              = special_dtype(vlen=bytes)
            ev              = np.array([(event.type, event.value) for event in events])
            filterBand      = [0, 60]
            processedData   = preproc.stdPreproc(data, filterBand, hdr)
            with File(file,'w') as f:
                # f.create_dataset('targets', data = tmp)
                f.create_dataset('rawData', data = data)
                f.create_dataset('events', data = ev, dtype = dt)
                f.create_dataset('processedData', data = processedData)
                # f.create_dataset()

            print("End calibration phase")

        elif e.value == "train":
            print("Training classifier")
            # linear detrend, filter, average across epochs
            # to be on the save side i picked a normal range;
            # there is some evidence that p300 is within the theta / alpha band.
            # Here i choose a wide band pass that would be able to catch it
            # and supress high frequency noise.
            model = classification.trainClassifier(file, hdr.fSample)
            bufhelp.sendEvent("training","done")

        elif e.value =="test":
            print("Feedback phase")
            keep = True
            while keep:
                data, events, stopevents = bufhelp.gatherdata(["stimulus"],\
                                                            trlen_ms,[("run","end"),\
                                                            ('test', 'end')], \
                                                            milliseconds=True, verbose = False)
                if stopevents.type == 'test' and stopevents.value == 'end':
                    keep = False
                else:
                    filterBand = [0, 60]
                    # linear detrend, filter, average across epochs
                    data = preproc.stdPreproc(data, filterBand, hdr)
                    pred = FeedbackSignals.genPrediction(data, model, events)
                    # sanity check
                    print('prediction', pred)
                    bufhelp.sendEvent("classifier.prediction", pred)

        elif e.value =="exit":
            run = False
        # print(e.value)
        print("Waiting for start event.")
