from __future__ import print_function, division

from pylab import *
import numpy as np
import scipy, scipy.signal


def plotERP(data, events, cap):
    '''
    Plots the ERP of the data per condition as indicated in events

    Inputs :
            Data = trials x time x channels
            Events = type x Value
    '''

    uniques = np.unique(events[:,1])                # find unique values of events

    # init mainframe and subplots
    nCols = 2 # default value to have at most 2 columns
    nRows = int(ceil(data.shape[-1] / nCols))
    print(nRows)
    fig, axes = subplots(nrows = nRows, ncols = nCols)
    mainFrame = fig.add_subplot(111, frameon = 0)
    mainFrame.tick_params(labelcolor = 'none',
    top='off', bottom='off', left='off', right='off')

    # for every channel print erp of conditions
    for unique in uniques:
        idx       = np.where(events[:, 1] == unique)[0] # get rows
        plotData  = np.mean(data[idx,...], 0 )
        for idx, ax in enumerate(axes.flatten()):
            ax.plot(plotData[..., idx], label = unique)
            ax.legend()
            try:
                ax.set_title(cap[idx, 0])
            except:
                continue

        # plot formatting
        subplots_adjust(hspace = .4)
        mainFrame.set_xlabel('Time[ms]')
        mainFrame.set_ylabel('mV')
    show()


def plotTF(data, events, cap = None):
    uniques = np.unique(events[:, 1])
    from matplotlib import ticker
    wavelets = np.logspace(0, 1.2, 25)
    convData = {}
    for unique in uniques:
        dataIdx = np.where(events[:, 1] == unique)[0]

        cw = scipy. signal.cwt(data[dataIdx, ...].flatten(),
                               scipy.signal.ricker,
                               wavelets)
        cw = cw.reshape(cw.shape[0], *data[dataIdx,...].shape)
        cw = np.mean(cw,0)
    #    convData[unique] = cw
        nCols = 2
        nRows = int(np.ceil(cw.shape[-1] / nCols))
        fig, axes = subplots(nrows = nRows, ncols = nCols)
        for idx, ax in enumerate(axes.flatten()):
            ax.imshow(cw[..., idx])

    show()





# def rickerWavelet(binnedData, nWavelet = 20, want = 'Pz', plotPos = 1,
#                 plotMin = -.4, plotMax = .4):
#     import seaborn as sb
#     from matplotlib import ticker
#     # sizeOfWavelets = logspace(.1, 1.31, nWavelet) # this goes to about 50 hz, more weighting on the lower end
#     sizeOfWavelets = linspace(.001, 15, 25)
#     convolutedData = {}
#     for type, data in binnedData.iteritems():
#         cw = scipy.signal.cwt(data.flatten(), scipy.signal.ricker, sizeOfWavelets)
#         cw = cw.reshape(cw.shape[0], data.shape[0], data.shape[1], data.shape[2])   # algorithm expects 1 d array
#         convolutedData[type] = cw
#         # print(cw.shape)                                           # reshape back
#
#     if plotPos:
#         # make figure for negative positive
#         fig, axs = subplots(5,2, sharex = 'all', sharey = 'all')
#         axx = fig.add_subplot(111,frameon = 0)
#         # mng         = get_current_fig_manager()
#         # mng.full_screen_toggle()
#         axx.grid('off')
#         for i, (ax, sensor) in enumerate(zip(axs.flatten(), cap)):
#             pos = convolutedData['positive'][..., i]
#             neg = convolutedData['negative'][..., i]
#             pos = mean(pos, 1)
#             neg = mean(neg, 1)
#             # print(pos.shape)
#             im  = ax.imshow( (pos - neg), origin = 'lower',
#                       extent = [0, 600, sizeOfWavelets[0], sizeOfWavelets[-1]],
#                       interpolation = 'None', aspect = 'auto', cmap = 'viridis',
#                       vmin = plotMin, vmax = plotMax)
#             # print(im)
#             tick = ticker.MaxNLocator(nbins = 4)
#             cc = colorbar(im, ax = ax)# cax = ax)
#             cc.set_label = 'Power'
#             cc.locator = tick
#             cc.update_ticks()
#             ax.grid('off')
#             ax.set_title(sensor[0])
#         subplots_adjust(hspace = .6)
#
#         fig.suptitle('Pos - neg')
#         xlabel('Time [ms]')
#         ylabel('Frequency [Hz]')
#         tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#         savefig('../Figures/feedback')
#         # show()
#
#     labels      = ['feet', 'left hand', 'right hand', 'rest']
#     titleLabels = ['Feet', 'Left Hand', 'Right hand', 'Rest']
#     for titleLabel, datasetLabel in zip(titleLabels, labels):
#         fig, axs = subplots(5,2)
#         mng         = get_current_fig_manager()              # full_screen_toggle
#         mng.full_screen_toggle()
#         fig.suptitle(titleLabel)
#         axx = fig.add_subplot(111, frameon = 0)
#         axx.grid('off')
#         dataset = convolutedData[datasetLabel]
#         dataset = np.mean(dataset, 1)
#         for i, (ax, sensor) in enumerate(zip(axs.flatten(), cap)):
#
#             im = ax.imshow(dataset[..., i],
#                     interpolation = 'bicubic',
#                     aspect = 'auto', origin = 'upper',
#                     extent = [0, 600, sizeOfWavelets[0], sizeOfWavelets[-1]],
#                     cmap = 'viridis')#, vmin = plotMin, vmax = plotMax)
#             ax.set_title(sensor[0])
#             ax.grid('off')
#             cc = colorbar(im, ax = ax)
#             cc.set_label('Power')
#             tick = ticker.MaxNLocator(nbins = 3)
#             cc.locator = tick
#             cc.update_ticks()
#         ax.set_title(sensor[0])
#
#         subplots_adjust(hspace = .6)
#         ylab = 'Frequency [Hz]'
#         xlab = 'Time[ms]'
#         xlabel(xlab, fontsize = 20)
#         ylabel(ylab, fontsize = 20)
#         tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#
#         sb.set_context('poster')
#         savefig('../Figures/Condition = ' + titleLabel)
#         # show()
#
#     # show particular electrode all locations [parsed as want]
#     electrode   = np.where(cap[:,0] == want)[0]
#     fig, ax     = subplots(2,2, sharex = 'all', sharey = 'all')
#            # full_screen_toggle
#
#     axx         = fig.add_subplot(111, frameon = False)
#     mng         = get_current_fig_manager()
#     mng.full_screen_toggle()
#     fig.suptitle(want)
#     axx.grid('off')
#     for axi, label, titleLabel in zip(ax.flatten(), labels, titleLabels):
#         dataset = convolutedData[label]
#         dataset = np.mean(dataset, 1)
#         # print(dataset[..., electrode].shape)d
#         im     = axi.imshow(dataset[..., electrode].squeeze(),
#         extent = [0, 1000, sizeOfWavelets[0], sizeOfWavelets[-1]], aspect = 'auto',
#         origin = 'lower', cmap = 'viridis',
#         interpolation = 'bicubic',
#         vmin = plotMin, vmax = plotMax)
#         sb.set_context('poster')
#         axi.set_title(titleLabel)
#         axi.grid('off')
#         tick = ticker.MaxNLocator(nbins = 3)
#         colorbar(im, ax = axi)
#         cc.locator = tick
#         cc.update_ticks()
#     xlabel('Time [ms]',      fontsize = 20)
#     ylabel('Frequency [Hz]', fontsize = 20)
#     axx.tick_params(axis = 'y', pad = 20)
#     sb.set_context('poster')
#     tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     savefig('../Figures/allConditions')
#     # show()
#         # assert 0


# def eventSeparator(data, events):
#     '''
#     Returns dictionary of key = event value, item = data corresponding to key
#     '''
#     uniqueEvents  = np.unique(events[:, 1])
#     eventStorage  = {}
#     for i, event in enumerate(uniqueEvents):
#         idx                   = np.where(events[:,1] == event)  # find corresponding indices
#         eventStorage[event]   = data[idx, :, :].squeeze()       # squeeze out the 0 dimension
#     return eventStorage

if __name__ == '__main__':
    from h5py import File
    with File('../Data/calibration_subject_MOCK_4.hdf5') as f:
        for i in f: print(i)
        procDataIM = f['procData/IM'].value
        eventsIM   = f['events/IM'].value

    print(procDataIM.shape)
    plotERP(procDataIM, eventsIM, cap = None)
    plotTF(procDataIM, eventsIM)
