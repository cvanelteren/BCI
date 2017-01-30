from h5py import File
import numpy as np
import preproc
import classification
import visualize
import os
from shutil import move
'''
This script edits the old experiment structure to fit with our final design
More specifically first we had events as one whole, i.e. it would contain
eventype x event value where all types where in the rows. The new design
separates the conditions and stores them in separate structure of the hdf5 file
'''
subjectNumber = 5
file = '../Data/calibration_subject_{0}.hdf5'.format(subjectNumber)
fileName = file.split('/')[-1]
print(fileName)

with File(file, 'r') as f:
    for i in f: print(i)

    events = f['events'].value
    rawData = f['rawData'].value
    cap  = f['cap'].value


uniques = np.unique(events[:, 0])
data = {}
ev   = {}
for unique in uniques:
    eventIdx = np.where(events[:, 0] == unique)
    data[unique] = rawData[eventIdx, ...].squeeze()
    ev[unique]   = events[eventIdx,...].squeeze()
dataIM = data['target']
eventsIM = ev['target']

eventsERN = ev['feedback']
dataERN = data['feedback']

procDataIM,  _  = preproc.stdPreproc(dataIM, [8, 20], 250)
procDataERN, _  = preproc.stdPreproc(dataERN,[5, 40], 250)

move(file, os.path.realpath('../Data/Backup/' + fileName + '.back'))

with File(file, 'w') as f :
    f.create_dataset('rawData/IM', data = dataIM)
    f.create_dataset('events/IM', data = eventsIM)
    f.create_dataset('procData/IM', data = procDataIM)

    f.create_dataset('events/ERN', data = eventsERN)
    f.create_dataset('rawData/ERN', data = dataERN)
    f.create_dataset('procData/ERN', data = procDataERN)

    f.create_dataset('fSample', data = 250)
    f.create_dataset('cap', data = cap)


# print(procDataIM.shape, eventsIM.shape)
# print(procDataERN.shape)
# visualize.plotERP(procDataIM, eventsIM, cap, fSample = 250)
# visualize.plotERP(procDataERN, eventsERN, cap, fSample = 250)
# classification.SVM(procDataIM[:,:-10,:], eventsIM, fft = 1)
# classification.SVM(procDataERN[:, :,:], eventsERN, fft = 0)
