from __future__ import division
import matplotlib
matplotlib.use('TkAgg')
from pylab import *
from subprocess import call
import subprocess
import sys
# sys.path.append('Project/Code/')


'''
Note to reader; there are some random numbers in this script; these are
mainly related to visual and thus can be tweaked to whatever we like
'''

print('Installing necessary modules...')
call(['python', 'moduleInstallers.py'])                 # check if all packages are installed

close('all')

fig, ax = subplots(1,1)
subplots_adjust(left=0, right=1, top=1, bottom=0)
# from matplotlib import pyplot as plt
# mng = get_current_fig_manager()
# mng.full_screen_toggle()
try:
    fig.canvas.toolbar.pack_forget()                     # remove toolbar
except:
    fig.canvas.toolbar = None                            # alt remove toolbar

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
    Connects with matplotlib figure and check for key press
    This is used for changing from calibration to test phase, see text
    Input : keypress event from figure
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

def startInTerminal(file):
    '''
    Uses subprocess.call to open a terminal an run a python script.
    Checks for possible versions depending on OS being run
    '''
    try:
        call(['powershell', 'python', file], creationflags = subprocess.CREATE_NEW_CONSOLE) #tested
    except:
        print('The OS is not windows, trying linux')
    try:
        call(['gnome-terminal','-x', 'python', file])                       #tested
    except:
        print('Gnome-terminal not detected, trying xterm')
    try:
        call(['xterm', '-e', file])                                         # untested
    except:
        print('xterm note detected, trying MAC terminal')
    try:
        call(['open', '-W', '-a', 'Terminal.app', 'python', file])          # tested
    except:
        print('Please check the details of the script, and edit the terminal you are using')
        raise UnknownOS



startInTerminal('signalProcessing.py')
def waitForKeyPress():
    '''
    Waits for a selection key
    Inputs:
            pressing 1 will initate calibration.py
            pressing 2 will initate cybathalonAdapter.py which will allow for playing the brainRunner game
            please note that running 2 will require running 1,
            or providing signalProcesing with a valid calibration file, see manual for more information
    '''
    global phase
    global returncode
    global running
    phase = 0
    alpha = True
    while alpha:
        if phase == 1:          # start calibration
            alpha = False
            startInTerminal('calibration.py')
        elif phase == 2:        # start brain runner interface
        	alpha = False
        	startInTerminal('cybathlonAdapter.py')
        elif phase == 3:        # close the mainframe
            alpha = False
            running = False
            returncode = 42
            plt.close(fig)
        pause(.1)

# set background
ax.set_facecolor('black')
# remove labels from axes
ax.set_xticks([])
ax.set_yticks([])

fig.canvas.mpl_connect('key_press_event', press)

# display welcome text
# create a figgure which is full screen in first place TODO
text = ax.text(0, 0,\
                 ('Welcome to the BCI Racer interface \n \n'
                'Press 1 to start the calibration\n'
                'Press 2 to start the race'),\
                 color      = 'white',
                 fontsize   = 20,\
                 horizontalalignment = 'center',\
                 verticalalignment = 'center')

global running
global returncode
running = True
while running:
    waitForKeyPress()
