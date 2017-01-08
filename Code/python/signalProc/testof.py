import numpy as np
from h5py import File
from pylab import *


with File('tmp.hdf5') as f:
    tmp = f['value'].value

print(tmp.shape)
fig, ax  = subplots(tmp.shape[-1])
for idx, i in enumerate(ax.flatten()):
    i.plot(np.sum(tmp[:,:,idx], 0))
fig.tight_layout()
show()
