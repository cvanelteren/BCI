from __future__ import division
import matplotlib
matplotlib.use('TkAgg')
from pylab import *

#import sys
#sys.path.append('../../../python/signalProc/')

# PARAMETERS
# prespecified number of targets
nTrials = 1 # 60
# timestep
tmp = .1
# controls how long to show a target (is multiplied with tmp)
target_duration = 20
# controls how long to show the feedback (is multiplied with tmp)
feedback_duration = 10
# controls how long to do nothing
rest_duration = 15
# proportion of negative feedback given
proportion_negative = 0.33
'''
Note to reader; there are some random numbers in this script; these are
mainly related to visual and thus can be tweaked to whatever we like
'''

# connect with the buffer
import bufhelp
fct, hdr = bufhelp.connect()

bufhelp.sendEvent('start', 'calibration')

close('all')

circleLabels = ['feet', 'right hand', 'left hand']

fig, ax     = subplots(1, 1)
subplots_adjust(left=0, right=1, top=1, bottom=0)
# from matplotlib import pyplot as plt
mng         = get_current_fig_manager()
mng.full_screen_toggle()
rcParams['toolbar'] = 'None'

r = 2
nCircle = len(circleLabels) + 1

# add one more for division in equal angles
angles = np.linspace(0, 2 * np.pi, nCircle)

# twice the r for plotting
rr = 2 * r
ax.set_xlim([-rr, rr])
ax.set_ylim([-rr, rr])


def press(event):
    '''
    looks for key press events
    '''
    # get the global start variablelen(capFile)  #
    global start
    # look for key event
    print('press', event.key)
    # spacebar is registered as ' ' for some reason
    if event.key == ' ':
        start = 1


def waitForSpacePress():
    '''
    Waits until space bar is pressed
    '''
    global start
    start = 0
    alpha = True
    while alpha:
        if start == 1:
            alpha = False
            text.set_visible(0)
        pause(.1)


# set background
ax.set_facecolor('black')
# remove labels from axes
ax.set_xticks([])
ax.set_yticks([])

# remove middle circle
nCond = len(circleLabels)

# create feedback vector
cutoff                      = nTrials * proportion_negative
give_feedback               = np.zeros((nTrials), dtype=int)
give_feedback[:int(cutoff)] = 1
np.random.shuffle(give_feedback)

# create target vectors
conditionLength = nTrials // nCond
targets = np.ones((nTrials, ), dtype=int)
# the idea here is to create 4 labels of equal size,
# later on arbitrarily i took 0 to be the false trials
# the rest will be correct  trials
for i in range(nCond):
    targets[i * conditionLength:(i + 1) * conditionLength] = i
# shuffle the target set
np.random.shuffle(targets)
fig.canvas.mpl_connect('key_press_event', press)

# display welcome text
# create a figure which is full screen in first place TODO
text = ax.text( 0,\
                0,\
                'Welcome\n Press space to start',\
                color = 'white',\
                horizontalalignment = 'center',\
                verticalalignment = 'center')

# colors
resetcolor      = '#404040'
targetcolor     = '#a6a6a6'
centercolor     = '#a6a6a6'
textcolor       = 'white'
restingcolor    = 'white'

waitForSpacePress()

# set up the circle objects
center      = (0, 0)
circles     = []
for i in range(nCircle):
    coordinate      = (np.sin(angles[i]), np.cos(angles[i]))
    # the last one is the one that moves
    if i == nCircle - 1:
        c = Circle(center, r / 8., color=resetcolor)
    else:
        c = Circle(coordinate, r / 4., color=resetcolor)
        ax.text(coordinate[0],\
                coordinate[1],\
                circleLabels[i],\
                color = textcolor,\
                horizontalalignment = 'center',\
                verticalalignment='center')
    circles.append(c)
    ax.add_artist(circles[i])

for idx, target in enumerate(targets):
    if (idx % 15 == 0) and idx > 0:
        ax.cla()
        # display break text
        # create a figure which is full screen in first place TODO
        text_str = 'Take a short break\n This was trial ' + \
                    str(idx) + ' of ' + str(len(targets)) + \
                    '\n Press space to start'

        text                = ax.text(0, 0,text_str,\
        color               = 'white',\
        horizontalalignment = 'center',\
        verticalalignment   = 'center')
        waitForSpacePress()

        # set up the circle objects
        center = (0, 0)
        circles = []
        for i in range(nCircle):
            coordinate = (np.sin(angles[i]), np.cos(angles[i]))
            # the last one is the one that moves
            if i == nCircle - 1:
                c = Circle(center, r / 8., color=resetcolor)
            else:
                c = Circle(coordinate, r / 4., color=resetcolor)
                ax.text(coordinate[0],\
                        coordinate[1],\
                        circleLabels[i],\
                        color = textcolor,\
                        horizontalalignment = 'center',\
                        verticalalignment   = 'center')
            circles.append(c)
            ax.add_artist(circles[i])

    # send event with target label to buffer'
    bufhelp.sendEvent('target', 'rest')
    # update color of circles
    circles[-1].center = center
    circles[-1].update({'color': restingcolor})
    circles[int(target)].update({'color': resetcolor})

    fig.canvas.draw()

    # run rest
    for i in range(rest_duration):
        fig.canvas.draw()
        pause(tmp)

    # send event with target label to buffer
    bufhelp.sendEvent('target', circleLabels[target])
    # update center from middle circle
    circles[-1].center = center
    circles[-1].update({'color': targetcolor})

    fig.canvas.draw()

    pause(tmp)
    circles[-1].update({'color': centercolor})
    circles[int(target)].update({'color': resetcolor})

    # run a target
    for i in range(target_duration):
        # Color target while animation
        circles[int(target)].update({'color': targetcolor})
        rand_move = np.random.randn() * 2 * np.pi
        coord = (.1 * np.sin(rand_move), .1 * np.cos(rand_move))
        circles[-1].center = coord
        fig.canvas.draw()
        pause(tmp)

    circles[-1].center = center
    circles[int(target)].update({'color': resetcolor})

    # show feedback
    # different feedback distribution needed? TODO

    if give_feedback[idx] == 1:
        circles[-1].update({'color': 'red'})
        # send event with feedback label to buffer
        bufhelp.sendEvent('feedback', 'negative')
    else:
        circles[-1].update({'color': 'green'})
        # send event with feedback label to buffer
        bufhelp.sendEvent('feedback', 'positive')

    fig.canvas.draw()
    pause(tmp * feedback_duration)
fig.clf()
bufhelp.sendEvent('calibration', 'end')

# train classifier
pause(1)
bufhelp.sendEvent('start', 'train')
# show()
close()
