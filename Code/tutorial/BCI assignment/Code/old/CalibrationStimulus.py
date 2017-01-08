from pylab import *
from bufferInterface import bufhelp
from experimentHelper import updateFigure, setupFigure, randomSequence

# PARAMETERS
state                   = 'calibration'
numOfCues               = 10
targetShowTime          = 2
waitForTrial            = 1
stimulusInterval        = .1
predictionShowTime      = 2


def press(event):
    '''
    looks for key press events in the figure
    '''
    # get the global start variable
    global start
    # look for key event
    # debug :
    # print('press', event.key)
    # spacebar is registered as ' ' for some reason
    if event.key == ' ':
        start = 1

# start checks if the user is ready for the experiment
global start
start = 0

fig, ax, text = setupFigure(state)
# connect with press event registry
fig.canvas.mpl_connect('key_press_event', press)

# connect to the buffer
ftc, hdr = bufhelp.connect()


# welcome the usershowPrediction
welcome_text = ('Press spacebar to continue')
updateFigure(text, welcome_text)

# wait for user to get ready
print('looking for user input')
lookingForInput = True
while lookingForInput:
    # without this pause while true is too fast
    pause(1)
    if start == 1:
        lookingForInput = False
        bufhelp.sendEvent('start', state)
        # debug:
        # print(state)

instruction_text = 'Think of your target letter \nGet ready'
sequences = [randomSequence() for _ in range(numOfCues)]
# targetShowTime = waitForTrial = stimulusInterval = 1e-4
# for every cue in the number of cues
for sequence in sequences:
    # run through the letters in the sequence
    for idx, stim in enumerate(sequence):
        # first case is target:
        if idx == 0:
            # send to buffer; but different label than normal targetstim
            # the subject is still focusing on the stimulus now, and
            # should not be included in the analysis of p300
            bufhelp.sendEvent('start target', stim)
            targetstim = stim
            # show target
            updateFigure(text, stim, target = 'target')
            # show target time
            pause(targetShowTime)
            # clear figure
            updateFigure(text, '')
            # wait time
            pause(waitForTrial)
        # run throught sequence
        else:
            updateFigure(text, stim)
            if  stim == targetstim:
                bufhelp.sendEvent('target', stim)
            else:
                bufhelp.sendEvent('stimulus', stim)
            pause(stimulusInterval)

bufhelp.sendEvent(state, "end")
pause(1)
# exit screen
exit_text = 'Thank you for participating!'
updateFigure(text, exit_text)

# train the classifier
bufhelp.sendEvent('start', 'train')
