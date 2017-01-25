from pylab import *
from numpy import *
import mne
import sklearn, sklearn.preprocessing
import scipy
from h5py import File
import preproc
num = 12
with File('../Data/calibration_subject_{0}.hdf5'.format(num)) as f:
    for i in f: print(i)
    # assert 0
    e = f['events'].value
    d = f['processedData'].value
    c = f['cap'].value
    d = f['test'].value
    try:
        t = f['test'].value

    except:
        pass

t = preproc.stdPreproc(t, [0, 40], 250)
print('done')
chanInfo = c[:,1:3]
# print(chanInfo.shape)

mD = mean(d.T, 2)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

fig, ax = subplots(5,2)
p = []
for axi in ax.flatten():
    p.append(axi.plot(linspace(0,600,150), random.rand(150) )[0])
    axi.set_ylim([-2,2])
t = t[:4000,:,:]
for i in range(t.shape[0]):
    for j in range(t.shape[-1]):
        # ax.flatten()[j].cla()
        p[j].set_ydata(t[i,:,j])


    # ax.plot(t[i,:,0])
    fig.canvas.draw()

    pause(1e-5)

# fig, ax = subplots()
# show()
# for i in range(10):
#     tmp, _ = mne.viz.plot_topomap(mD[:,i],chanInfo, show = False, names =  c[:,0], sensors = False, show_names=True)
#     # ax.tmp
#     draw()
