#!/usr/bin/python
#rewritten script in python from the matlab adaptation
#  Author:  Casper van Elteren

import bufhelp
import pickle
import numpy as np
import classification as clsf
import preproc
from h5py import File, special_dtype
from pylab import *
ftc, hdr = bufhelp.connect()
file = 'test_script.hdf5'
trlen_ms = 1000
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

            data, events, stopevents = bufhelp.gatherdata(\
            ['target','stimulus'], \
            trlen_ms, ("calibration", "end"), \
            milliseconds=True, verbose = False)
            data = np.array(data)
            dt = special_dtype(vlen=bytes)
            ev = np.array([(event.type, event.value) for event in events])
            print(data.shape, ev.shape)
            with File(file,'w') as f:
                # f.create_dataset('targets', data = tmp)
                f.create_dataset('data', data = data)
                f.create_dataset('events', data = ev, dtype = dt)
                # f.create_dataset()

            print("End calibration phase")

        elif e.value == "train":
            print("Training classifier")
            data = preproc.detrend(data)
            useable = preproc.badChannelRemoval(data)
            data = data[:,:, useable]
            data = preproc.butter_filter(data,[0, 40], hdr = hdr)
            data, events = preproc.formatData(data, events)
            clsf = clsf.linear_svm(data, events)
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

                data = preproc.detrend(data)
                useable = preproc.badChannelRemoval(data)
                data = data[:,:, useable]
                data = preproc.butter_filter(data,[0, 40], hdr = hdr)
                data, _ = preproc.formatData(data, events)

                predictions = clsf.predict(data)
                x = np.where(predictions == 1)
                if x == []:
                    predictions = 'dunno'
                else:
                    predictions = events[x].value
                # fig, ax = subplots(1,1)
                # ax.plot(predictions)
                # show()
                # print(predictions)
                bufhelp.sendEvent("classifier.prediction",predictions)

        elif e.value =="exit":
            run = False
        # print(e.value)
        print("Waiting for start event.")
