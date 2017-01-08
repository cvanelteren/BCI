from h5py import File
from pylab import *
from numpy import *


with File('test.mat','r') as f :


    d = f['data']['buf']
    tmp = d[0][0]
    print(tmp[0])
