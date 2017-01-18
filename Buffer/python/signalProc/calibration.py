from __future__ import division
from pylab import *
#import sys
#sys.path.append('../../../python/signalProc/')

# connect with the buffer
import bufhelp
fct, hdr = bufhelp.connect()

close('all')

circleLabels = [ 'feet' , 'right hand', 'left hand']

fig, ax = subplots(1,1)

r = 2
nCircle = len(circleLabels)  + 1

# add one more for division in equal angles
angles = np.linspace(0, 2 * np.pi, nCircle)

# twice the r for plotting
rr = 2 * r
ax.set_xlim([-rr,rr])
ax.set_ylim([-rr,rr])




def press(event):
    '''
    looks for key press events
    '''
    # get the global start variable
    global start
    # look for key event
    print('press', event.key)
    # spacebar is registered as ' ' for some reason
    if event.key == ' ':
        start = 1

# set background
ax.set_facecolor('black')
# remove labels from axes
ax.set_xticks([])
ax.set_yticks([])


# prespecified number of targets
nTrials = 5
# remove middle circle
nCond = len(circleLabels)

# create target vectors
conditionLength = nTrials // nCond
targets = np.ones((nTrials,),dtype = int)
for i in range(nCond):
    targets[i * conditionLength: (i + 1) * conditionLength] = i
# shuffle the target set
np.random.shuffle(targets)
fig.canvas.mpl_connect('key_press_event', press)

text = ax.text(0,0, 'Welcome\n Press space to start', color = 'white', horizontalalignment = 'center', verticalalignment = 'center')

# colors
resetcolor = 'gray'
targetcolor = 'slategray'

start = 0
alpha = True
while alpha:
    if start == 1:
        alpha = False
        text.set_visible(0)
    pause(.1)

# set up the circle objects
center =  (0,0)
circles = []
for i in range(nCircle):
    coordinate = (np.sin(angles[i]), np.cos(angles[i]))
    # the last one is the one that moves
    if i == nCircle-1:
        c = Circle(center, r / 8., color = resetcolor)
    else:
        c = Circle(coordinate, r / 4., color = resetcolor)
        ax.text(coordinate[0], coordinate[1], circleLabels[i],\
        color = 'white',  horizontalalignment='center', verticalalignment='center')
    circles.append(c)
    ax.add_artist(circles[i])

# targets
tmp = .1
target_duration = 10
for target in targets:
    # send event to buffer
    bufhelp.sendEvent('target', circleLabels[target] )
    # update center from middle circle
    circles[-1].center = center
    circles[-1].update({'color': targetcolor})
    circles[int(target)].update({'color':targetcolor})

    fig.canvas.draw()

    pause(tmp)
    circles[-1].update({'color':'white'})
    circles[int(target)].update({'color': resetcolor})

    # run a target
    for i in range(target_duration):
        rand_move = np.random.randn()*2*np.pi
        coord = (.1 * np.sin(rand_move), .1 *np.cos(rand_move))
        circles[-1].center = coord
        fig.canvas.draw()
        pause(tmp)

    circles[-1].center = center
    # show  right wrong
    if target != 0:
        circles[-1].update({'color':'red'})
    else:
        circles[-1].update({'color':'green'})
    fig.canvas.draw()
    pause(tmp)
fig.clf()
bufhelp.sendEvent('stimulus.training','end')
show()
