import time
from pylab import *
from bufferInterface import bufhelp
from experimentHelper import updateFigure, setupFigure, randomSequence


# PARAMETERS
state                   = 'test'
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
    # present instructions
    # show the instruction text
    updateFigure(text, instruction_text)
    # w ait for x seconds
    pause(targetShowTime)
    # clear the figure
    updateFigure(text, '')
    # wait for trial to start
    pause(waitForTrial)

    # run through the letters in the sequence
    for idx, stim in enumerate(sequence):
        # skip the first cases
        if idx != 0:
            updateFigure(text, stim)
            bufhelp.sendEvent('stimulus', stim)
            pause(stimulusInterval)

    # show signalprocessing that run has ended
    bufhelp.sendEvent('run', 'end')
    # after the runs clear the figure
    updateFigure(text, ' ')
    # wait at most targetShowTime
    tic = time.time()
    event = bufhelp.waitforevent("classifier.prediction", targetShowTime* 1000)
    # in case of timer running out show message of failure
    if event == None:
        pred = 'Classification failed'
    else:
        pred = event.value
    elap = time.time() - tic
    # print('> elapsed', elap)
    if elap > 1:
        elap = elap - 1
    else:
        pause(waitForTrial - elap)
        elap = 0
    # wait
    pause(predictionShowTime)
    bufhelp.sendEvent('prediction', pred)
    updateFigure(text, pred, target = 'prediction')
    # print('elapse', elap)
    # normalize the wait time
    pause(targetShowTime - elap)

bufhelp.sendEvent(state, "end")
pause(1)
# exit screen
exitText = 'Thank you for participating!'
updateFigure(text, exitText)

bufhelp.sendEvent('start', 'exit')
