bufferpath = "../../python/signalProc"

import os, sys, random, math, time, socket, struct
sys.path.append(os.path.dirname(__file__)+bufferpath)
import bufhelp
from pylab import *

close('all')

def press(event):
    '''
    looks for key press events
    '''
    # get the global player variable
    global player
    # look for key event
    print('press', event.key)
    # spacebar is registered as ' ' for some reason
    if event.key == '1':
        player = 1
    elif event.key == '2':
        player = 2
    elif event.key == '3':
        player = 3
    elif event.key == '4':
        player = 4

def waitForKeyPress():
    global player
    player = 0
    alpha = True
    while alpha:
        if player != 0:
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

text = ax.text(0, 0,\
'Please choose the player you are using:\n Press 1 for Player 1\n Press 2 for Player 2\n Press 3 for Player 3\n Press 4 for Player 4',\
color = 'white',\
horizontalalignment = 'center',\
verticalalignment = 'center')

waitForKeyPress()

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
br_player=1

global second_pred
second_pred = None

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
CMDS      = [CMD_ROLL, CMD_RST, CMD_JUMP, CMD_SPEED]
THRESHOLDS= [.1,        .1,       .1,     .1      ]

# Sends a command to BrainRacer.
def send_command(command):
	global br_socket
	print("Send cmd " + str(command) )
	cmd = (br_player * 10) + command
	data = struct.pack('B', cmd)
	
	br_socket.sendto(data, (br_hostname, br_port))
	
#Connect to BrainRacers
br_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);

#Connect to Buffer
(ftc,hdr) = bufhelp.connect(buffer_hostname,buffer_port)
print("Connected to " + buffer_hostname + ":" + str(buffer_port))
print(hdr)

def max2(numbers):
    i1 = i2 = None
    m1 = m2 = float('-inf')
    for i,v in enumerate(numbers):
        if v > m2:
            if v >= m1:
                m1, m2 = v, m1
                i1, i2 = i, i1
            else:
                m2 = v
                i2 = i
    return ([m1,m2],[i1,i2])

# Receive events from the buffer and process them.
def processBufferEvents():
	global running
	events = bufhelp.buffer_newevents()

	for evt in events:
		print(str(evt.sample) + ": " + str(evt))

		if evt.type == 'clsfr.prediction.im':
			pred = evt.value
			(m12,i12) = max2(pred) # find max value
			second_pred = i12[1]
			if m12[0]-m12[1] > THRESHOLDS[i12[0]] : send_command(CMDS[i12[0]]); # if above threshold send
		
		elif evt.type == 'clsfr.prediction.ern':
			pred = evt.value
			(m12,i12) = max2(pred) # find max value
			# TODO
			# valid prediction of ERN
			if m12[0]-m12[1] > 0.1:
				send_command(CMDS[second_pred]);

		elif evt.type == 'keyboard':
			if   evt.value == 'q' :  send_command(CMD_SPEED)
			elif evt.value == 'w' :  send_command(CMD_JUMP)
			elif evt.value == 'e' :  send_command(CMD_ROLL)
			elif evt.value == 'esc': running=false

		elif evt.type == 'startPhase.cmd':
			if evt.value == 'quit':
				running = False


# Receive events until we stop.	
running = True
while running:
	processBufferEvents()

