from __future__ import division
import matplotlib
matplotlib.use('TkAgg')
from pylab import *
from subprocess import call



'''
Note to reader; there are some random numbers in this script; these are
mainly related to visual and thus can be tweaked to whatever we like
'''


close('all')

figg, ax = subplots(1,1)
subplots_adjust(left=0, right=1, top=1, bottom=0)
# from matplotlib import pyplot as plt
# mng = get_current_fig_manager()
# mng.full_screen_toggle()

r = 2
nCircle = 4

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
    global phase
    # look for key event
    print('press', event.key)
    # spacebar is registered as ' ' for some reason
    if event.key == '1':
        phase = 1
    elif event.key == '2':
    	phase = 2
    elif event.key == 'escape':
        phase = 3

def waitForKeyPress():
    global phase
    global returncode
    global running
    phase = 0
    alpha = True
    while alpha:
        if phase == 1:
            alpha = False
            returncode = call(['python','Project/Code/calibration.py'])
        elif phase == 2:
        	alpha = False
        	returncode = call(['python','Project/Code/CybathlonAdapter.py'])
        elif phase == 3:
            alpha = False
            running = False
            returncode = 42
            plt.close(figg)
        pause(.1)

# set background
ax.set_facecolor('black')
# remove labels from axes
ax.set_xticks([])
ax.set_yticks([])

figg.canvas.mpl_connect('key_press_event', press)

# display welcome text
# create a figgure which is full screen in first place TODO
text = ax.text(0, 0,\
'Welcome to the BCI Racer interface \n Press 1 to start the calibration\n Press 2 to start the race',\
color = 'white',\
horizontalalignment = 'center',\
verticalalignment = 'center')

global running
global returncode
running = True
while running:
    waitForKeyPress()
