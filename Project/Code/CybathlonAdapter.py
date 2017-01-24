bufferpath = "../../python/signalProc"

import os, sys, random, math, time, socket, struct
sys.path.append(os.path.dirname(__file__)+bufferpath)
import bufhelp
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from pylab import *
import pickle

close('all')

def press(event):
    '''
    looks for key press events
    '''
    # get the global choice variable
    global choice
    # look for key event
    print('press', event.key)
    if event.key == '1':
        choice = 1
    elif event.key == '2':
        choice = 2
    elif event.key == '3':
        choice = 3
    elif event.key == '4':
        choice = 4
    elif event.key == ' ':
    	choice = 5

def waitForKeyPress():
    global choice
    choice = 0
    alpha = True
    while alpha:
        if choice != 0:
        	break
        pause(.1)


fig, ax = subplots(1,1)
subplots_adjust(left=0, right=1, top=1, bottom=0)

r = 2
nCircle = 4

# add one more for division in equal angles
angles = np.linspace(0, 2 * np.pi, nCircle)

# twice the r for plotting
rr = 2 * r
ax.set_xlim([-rr,rr])
ax.set_ylim([-rr,rr])

mng = get_current_fig_manager()
mng.full_screen_toggle()

# set background
ax.set_facecolor('black')
# remove labels from axes
ax.set_xticks([])
ax.set_yticks([])

fig.canvas.mpl_connect('key_press_event', press)

while ((choice < 1) or (choice > 4)):
	text = ax.text(0, 0,\
	'Please choose the player you are using:\n Press 1 for Player 1\n Press 2 for Player 2\n Press 3 for Player 3\n Press 4 for Player 4',\
	color = 'white',\
	horizontalalignment = 'center',\
	verticalalignment = 'center')

	waitForKeyPress()

# Save player number
player = choice
choice = 0
ax.cla()

while ((choice < 1) or (choice > 2)):
	text_str = 'Which version do you want to play?\n Press 1 for Imagined Movement \n Press 2 for Imagined Movement Plus'

	text                = ax.text(0, 0,text_str,\
	color               = 'white',\
	horizontalalignment = 'center',\
	verticalalignment   = 'center')

	waitForKeyPress()

# Save version number
version = choice
choice = 0
ax.cla()

# Command offsets, do not change.
if player == 1:
	CMD_SPEED= 1
	CMD_JUMP = 2
	CMD_ROLL = 3
elif player ==2:
	CMD_SPEED= 11
	CMD_JUMP = 12
	CMD_ROLL = 13
elif player ==3:
	CMD_SPEED= 21
	CMD_JUMP = 22
	CMD_ROLL = 23
elif player ==4:
	CMD_SPEED= 31
	CMD_JUMP = 32
	CMD_ROLL = 33
CMD_RST  = 99 # if sth unrecognizable is sent, it will ignore it

# Command configuration
CMDS      = [CMD_ROLL, CMD_JUMP, CMD_SPEED, CMD_RST]
verbCMDS  = ['Roll','Jump','Speed','Rest']
# THRESHOLDS= [.1,        .1,       .1,     .1      ]
THRESHOLDS= [.025,        .025,       .025,     .025      ]

# Load dictionaries containing int to label mapping for movement and label to int mapping for ern
# i2l = pickle.load(open('Project/Code/i2l_im.pkl','rb'))
# l2i = pickle.load(open('Project/Code/l2i_ern.pkl','rb'))
labels = sort(['left hand','right hand', 'feet', 'rest'])
# Probably introduce instructions here
while (choice < 5):
	text_str =\
    'Instructions: \n\n ' + \
            verbCMDS[0] + ' command will executed when thinking about your ' + labels[0] + \
	'\n ' + verbCMDS[1] + ' command will executed when thinking about your ' + labels[1] + \
	'\n ' + verbCMDS[2] + ' command will executed when thinking about your ' + labels[2] + \
	'\n ' + verbCMDS[3] + ' command will executed when thinking about your ' + labels[3] + \
	'\n\n Press space to start'

	text                = ax.text(0, 0,text_str,\
	color               = 'white',\
	horizontalalignment = 'center',\
	verticalalignment   = 'center',\
    fontsize            =  20 )

	waitForKeyPress()
choice = 0
ax.cla()

fig.canvas.draw()
fig.clf()
pause(1)
close(fig)

# Configuration of buffer
buffer_hostname='localhost'
buffer_port=1972

# Configuration of BrainRacer
br_hostname='localhost'
br_port=5555
br_player=player

global second_pred
second_pred = None

# Sends a command to BrainRacer.
def send_command(command, v = verbCMDS):
	global br_socket

	# print("Send cmd " + str(v[command]), command ) # printing cant keep up and will error
	cmd = (br_player * 10) + command
	data = struct.pack('B', cmd)

	br_socket.sendto(data, (br_hostname, br_port))

#Connect to BrainRacers
br_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);

#Connect to Buffer
(ftc,hdr) = bufhelp.connect(buffer_hostname,buffer_port)
print("Connected to " + buffer_hostname + ":" + str(buffer_port))
print(hdr)

print(version)

bufhelp.sendEvent('start', 'test')
# Receive events from the buffer and process them.
# Receive events until we stop.
running = True
while running:
    global running
    events = bufhelp.buffer_newevents()
    nChans = 0
    for evt in events:
    	# print( str(evt.sample) + ": " + str(evt) )
    	if evt.type == 'clsfr.prediction.im' :
            pred = evt.value
            bestIdx = np.argmax(pred)
            if pred[bestIdx] > 1 / float(len(pred)):
                send_command(CMDS[bestIdx])
            else:
                send_command(np.random.randint(0, len(pred)))
    	elif evt.type == 'clsfr.prediction.ern' and version == 2:
            pred = evt.value
            bestIdx = np.argsort(pred)[-2]
            if pred[bestIdx] > 1 / (float( len (pred) ) ):
                send_command(CMDS[bestIdx])
            else:
                send_command(np.random.randint(0, len(pred)))

    	elif evt.type == 'keyboard':
    		if   evt.value == 'q' :  send_command(CMD_SPEED)
    		elif evt.value == 'w' :  send_command(CMD_JUMP)
    		elif evt.value == 'e' :  send_command(CMD_ROLL)
    		elif evt.value == 'esc': running=false

    	elif evt.type == 'test':
    		if evt.value == 'end':
    			running = False



	# processBufferEvents()
