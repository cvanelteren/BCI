from numpy import *
from pylab import *
from h5py import File

with File('calibration_subject_0.hdf5') as f:
    for i in f: print(i)
    e = f['events'].value

print(e)
