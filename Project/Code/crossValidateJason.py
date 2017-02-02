from h5py import File
import numpy as np
from scipy.io import loadmat

subjectData = loadmat('../Data/example_test.mat')

devent  = subjectData['devents']
data    = subjectData['data']

print(subjectData.keys())
tmp = subjectData['hdr']
print(tmp)
# print(data[0][0][0].shape)
tmpData = []
tmpEvent =[]
for i, j in zip(data, devent):
    tmpData.append(i[0][0].T)
    j = j[0]
    # print(j.shape, j[0], j[-1]); assert 0
    # if j[-1]   == 1 :
    #     setValue = 'target 1'
    # elif j[-1] == 2:
    #     setValue = 'target 2'
    # elif j[-1] == 3:
    #     setValue = 'target 3'
    # else :
    #     setValue = 'target 0'
    setValue = str(j[-1][0])
    tmpEvent.append(['target',setValue])


tmpdata = np.array(tmpData, dtype = float)

tmpevent = np.array(tmpEvent, dtype = str)
print(tmpdata.shape, tmpevent.shape)
print(tmpevent.dtype, tmpdata.dtype)
with File('../Data/transcodes72.hdf5', 'w') as f :
    f.create_dataset('rawData/IM', data  = tmpdata)
    f.create_dataset('events/IM', data    = tmpevent)
    # for i in f:print(i)

import classification, preproc

p, _  = preproc.stdPreproc(tmpData, [8, 20], 100)
# classification.SVM(p, tmpevent)
