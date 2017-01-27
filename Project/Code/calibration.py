from __future__ import division
import matplotlib
matplotlib.use('TkAgg')     # In case python doesn't default to this
from pylab import *

'''
Instructions :
    This script will display 4 circles:
        Three indicating the imagined movement (feet, left and right hand)
        The center circle will move during a trial run
        After a trial is completed the center circle will indicate whether the
        imagined movement was correct, note that this is not related to a classifier
        and is purely false feedback.
        Parameters with respect to timing can be set below

    At the end this script will send a train classifier event
'''

# PARAMETERS
# prespecified number of targets
nTrials             = 60
dt                  = .1              # multiplication factor (time step)
targetDuration      = 20              # target show time
feedbackDuration    = 10              # feedback show time
restDuration        = 15              # duration of rest
proportionNegative  = 1/3             # proportion of negative feedback
breakTrial          = 120              # break after x trials


# connect with the buffer
import bufhelp
fct, hdr = bufhelp.connect()


close('all')

circleLabels = ['feet', 'right hand', 'left hand']

# open a figure remove the toolbar
fig, ax     = subplots(1, 1, facecolor = 'black')
subplots_adjust(left   = 0,
                right  = 1,
                top    = 1,
                bottom = 0)                          # full screen [mac users]
mng         = get_current_fig_manager()              # full_screen_toggle
mng.full_screen_toggle()

ax.set_aspect('equal')
# set background
ax.set_facecolor('black')
# remove labels from axes
ax.set_xticks([])
ax.set_yticks([])

# this errors in ipython for some reason, not from terminal
try:
    fig.canvas.toolbar.pack_forget()                     # remove toolbar
except:
    fig.canvas.toolbar = None                            # alt remove toolbar

r       = 10                    # radius
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
    # print('press', event.key)
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



# remove middle circle
nCond = len(circleLabels)

# STIMULUS VECTOR
# create feedback vector
cutoff                      = round(nTrials * proportionNegative)
give_feedback               = np.zeros((nTrials), dtype=int)
give_feedback[:int(cutoff)] = 1
np.random.shuffle(give_feedback)

# create target vectors
conditionLength = nTrials // nCond
targets         = np.ones((nTrials, ), dtype = int)
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
                verticalalignment = 'center', fontsize = 20)

# colors
resetcolor      = '#404040'
targetcolor     = '#a6a6a6'
centercolor     = '#a6a6a6'
textcolor       = 'white'
restingcolor    = 'white'

waitForSpacePress()

# SETUP OF CIRCLES : (action wheel)
center      = (0, 0)
# set up the circle objects
circles     = []
texts       = []
for i in range(nCircle):
    # this is the center circle
    coordinate      = np.array([
                        np.sin(angles[i]),
                        np.cos(angles[i])]) * 6 # magic number is length scaling
    # The moving center
    if i == nCircle - 1:
        c = Circle(center, r / 12., color=resetcolor)     # magic number was subjective
    # the imagined movement nodes
    else:
        c       = Circle(coordinate, r / 3., color=resetcolor)  # magic number was subjective
        text    = ax.text(coordinate[0],\
                coordinate[1],\
                circleLabels[i],\
                color    = textcolor,\
                horizontalalignment = 'center',\
                verticalalignment   = 'center', \
                fontsize = 20)
    circles.append(c)
    texts.append(text)
    ax.add_artist(circles[i])


# display break text
# create a figure which is full screen in first place TODO


textBreak           = ax.text(0, 0,'',\
color               = 'white',\
horizontalalignment = 'center',\
verticalalignment   = 'center', \
fontsize            = 20)

bufhelp.sendEvent('start', 'calibration')
for idx, target in enumerate(targets):
    # every break trials; take a break wait for user input
    if (idx % breakTrial == 0) and idx > 0:
        for c, text in zip(circles, texts):
            c.set_visible(False)
            text.set_visible(False)
        fig.canvas.draw()
        text_str = 'Take a short break\n This was trial ' + \
                    str(idx) + ' of ' + str(len(targets)) + \
                    '\n Press space to start'
        textBreak.set_text(text_str)
        textBreak.set_visible(True)             # show break text
        waitForSpacePress()                     # wait for user input
        textBreak.set_visible(False)            # remove break text
        for c, text in zip(circles, texts):
            c.set_visible(True)
            text.set_visible(True)

    bufhelp.sendEvent('target', 'rest')         # buffer event

    # update color of circles
    circles[-1].center = center
    circles[-1].update({'color': restingcolor})
    circles[int(target)].update({'color': resetcolor})

    fig.canvas.draw()

    pause(restDuration * dt)                # rest trial

    bufhelp.sendEvent('target',             # send event to buffer
                    circleLabels[target])
    circles[-1].center = center             # move the center circle to center

    # SHOW FEEDBACK
    circles[-1].update({'color': targetcolor})

    fig.canvas.draw()

    pause(dt)
    circles[-1].update({'color': centercolor})
    circles[int(target)].update({'color': resetcolor})

    # run a target
    for i in range(targetDuration):
        # Color target while animation
        circles[int(target)].update({'color': targetcolor})
        rand_move = np.random.randn() * 2 * np.pi
        coord = (.1 * np.sin(rand_move), .1 * np.cos(rand_move))
        circles[-1].center = coord
        fig.canvas.draw()
        pause(dt)

    # SHOW FEEDBACK
    circles[-1].center = center
    circles[int(target)].update({'color': resetcolor})

    # Positive trials
    if give_feedback[idx] == 1:
        circles[-1].update({'color': 'red'})
        # send event with feedback label to buffer
        bufhelp.sendEvent('feedback', 'negative')
    # Negative trials
    else:
        circles[-1].update({'color': 'green'})
        # send event with feedback label to buffer
        bufhelp.sendEvent('feedback', 'positive')

    # circles[-1].center = (np.sin(angles[target]), np.cos(angles[target]))
    fig.canvas.draw()
    pause(dt * feedbackDuration)
fig.clf()
bufhelp.sendEvent('calibration', 'end')

# train classifier
pause(1)
bufhelp.sendEvent('start', 'train')
# show()
close()
