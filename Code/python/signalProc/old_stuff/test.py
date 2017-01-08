from pylab import *
import sys
sys.path.append('../../../python/signalProc/')

# connect with the buffer

fct, hdr = bufhelp.connect()


fig, ax = subplots(1,1)

r = 2
nCircle = 4

# add one more for division in equal angles
angles = np.linspace(0, 2 * np.pi, nCircle)

print(angles)
# twice the r for plotting
rr = 2 * r
ax.set_xlim([-rr,rr])
ax.set_ylim([-rr,rr])

# set up the circle objects
center =  (0,0)
circles = []
for i in range(nCircle):
    coordinate = (np.sin(angles[i]), np.cos(angles[i]))
    # the last one is the one that moves
    if i == nCircle-1:
        c = Circle(center, r / 8, color = 'gray')
    else:
        c = Circle(coordinate, r/4, color = 'gray')

    circles.append(c)
    ax.add_artist(circles[i])

# set background
ax.set_facecolor('k')
# remove labels from axes
ax.set_xticks([])
ax.set_yticks([])

nTrials = 100
nCond = 4
conditionLength = nTrials // nCond
 
targets = np.ones((nTrials,))
for i in range(nCond):
    targets[i * conditionLength: (i + 1) * conditionLength] = i


## trials
#trial_start_time = 1
#trial_duration = 10
#for trial in range(5):
#    
#    buffhelp.sendEvent('target','value')
#    circles[-1].center = center
#    circles[-1].update({'color':'green'})
#    fig.canvas.draw()
#    pause(trial_start_time)
#    circles[-1].update({'color':'white'})
#    for i in range(trial_duration):
#        rand_move = np.random.randn()*2*np.pi
#        coord = (.1 * np.sin(rand_move), .1 *np.cos(rand_move))
#        circles[-1].center = coord
#        fig.canvas.draw()
#        pause(1)
#        
#    # show  right wrong
#    if np.random.rand() < 1/3:
#        circles[-1].center = center
#        circles[-1].update({'color':'magenta'})
#        fig.canvas.draw()
#    pause(1)
#    
#    
#
#show()
#
#close('all')




# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
# text = ax.text(\
#         0.5*(left+right),\
#         0.5*(bottom+top), '',
#         horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=20, color='white',
#         transform=ax.transAxes)
# text.set_text('test')
#
#
# r = 1
#
# print(c)
#
# show()
