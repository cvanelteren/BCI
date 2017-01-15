
""" Allows for looking at the incomming raw data"""
from pylab import *
from numpy import *
from bufhelp import *

ftc, hdr = connect()
nChans = hdr.nChannels
dt =  1 / hdr.fSample

# PARAMETERS
plotTime = 2
nPoints  = int(plotTime / dt)
time = linspace(-nPoints, 0, nPoints)
bufferStorage = zeros((nChans, nPoints))

# square tile plot if we have even channels otherwise more rows than columns
if nChans % 2  == 0:
    nRows = nCols = nChans // 2
else:
    nRows = (nChans + 1) // 2
    nCols = nChans // nRows

print time.shape, bufferStorage.shape
# init figure
fig, axes = subplots(nrows = nRows, ncols = nCols, sharex = 'all')
xlab =  'Time lag (step)'
# print(1/(len(xlab)/2.)); assert 0
fig.text(.5 - 1/(len(xlab)/2.), .04, xlab)
p = [ax.plot(time, bufferStorage[0,:]) for ax in axes.flatten()]


while True:
    # get latest samples and plot them
    idx = ftc.getHeader().nSamples  - 1
    lastSample =  ftc.getData((idx,idx))
    # shift the storage
    bufferStorage = roll(bufferStorage, -1)
    bufferStorage[:, -1] = lastSample
    print lastSample
    for i, pi in enumerate(p):
        pi = pi[0] # for some reason it is a list with 1 object
        chanData = bufferStorage[i,:]
        minChanData = min(chanData)
        maxChanData = max(chanData)
        pi.set_ydata(chanData)
        axes.flatten()[i].set_ylim([minChanData,maxChanData])
        draw()
    pause(1e-1)
