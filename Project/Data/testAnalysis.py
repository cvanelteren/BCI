from numpy import *
from pylab import *
import mne
from h5py import File

with File('calibration_subject_4.hdf5') as f:
    for i in f: print(i)
    e = f['events'].value
    d = f['processedData'].value
    c = f['cap'].value


chanInfo = c[:,1:3]
# print(chanInfo.shape)

mD = mean(d.T, 2)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
fig, ax = subplots()
show()
for i in range(10):
    tmp, _ = mne.viz.plot_topomap(mD[:,i],chanInfo, show = False, names =  c[:,0], sensors = False, show_names=True)
    # ax.tmp
    draw()
