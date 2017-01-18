from pylab import *

import matplotlib.animation as animation
import string
import numpy as np
# matplotlib.use('TkAgg')git clone https://github.com/lundal/vibou.gTile.git ~/.local/share/gnome-shell/extensions/gTile@vibou

# check matplotlib version
# we need to have the latest versions otherwise some modules are not
# loading properly    print(len(sequence[0]))

import sys
sys.path.append('../../')
if not matplotlib.__version__[0] == '2':
    import pip
    pip.main(['install', 'matplotlib==2.0.0b4'])

def experiment(state):
    '''
    Runs the experiment described in the pdf, either calibration or test phase
    '''

    # default 5
    def random_sequence(n_repeats = 5):
        '''
        The order of the letters is randomized, this is interpreted as
        shuffling the alphabet for n_repeat times, i.e. 5 totally random sequences


        returns a random sequence with target letter
        the sequence consists of 1 target letter and 5 repeats of the entire
        alphabet
        '''
        # only upper case letters
        letters = string.ascii_uppercase
        alphabet = [x for x in letters]
        randomOrder = np.random.randint(0,len(letters))
        targetLetter = alphabet[randomOrder]
        sequence = targetLetter
        for i in range(n_repeats):
            # random shuffle
            np.random.shuffle(alphabet)
            # stack the strings
            sequence = np.hstack((sequence, alphabet))
        return sequence


    def update_figure(text, letter, target = 'stimulus'):
        '''
        Updates the figure with whatever is in letter
        defaults to stimulus
        '''

        text.set_text(letter)
        # target is green
        if target   == 'target':
            text.set_color('green')
        # blue for testing phase, shows the prediction
        elif target == 'prediction':
            text.set_color('blue')
        # within the trial we have white letters
        else:
            text.set_color('white')


    # this can be used for blitting but is not necessary

    # def calibration_phase():
    #     '''
    #     To do:
    #         get ready and space bar before starting
    #     '''
    #     ani = animation.FuncAnimation(fig, run_experiment, blit = 0, fargs = ('calibration',))
    #     return ani
    #
    # def test_phase():
    #     # run the experiment
    #     ani = animatixon.FuncAnimation(fig, run_experiment, blit = 1, fargs = ('test',))
    #     return ani


    # num of cues default 10
    # the i index if for blitting options above, but i removed those for
    # compatibility reasons
    def run_experiment(i, state, num_of_cues = 10,  target_show_time  = 2, wait_for_trial = 1,\
                        interval_stimulus = .1,\
                        show_prediction = 2
                        ):
        '''
        This runs the basic experiment
        The default parameters set up the amount of loops and the timing between
        stimulus presentation

        The basic experiment consists of :
            showing target (10 times)
                repeating 5 loops of alphabet sequence (order randomized)

        '''
        # print(num_of_cues, target_show_time)
        # print ('in the zone')
        # if test phase show welcome message

        # if state == 'test':
        # lookingstopeventruns for start indicator
        global start

        # welcome the user
        welcome_text = ('Press spacebar to continue')

        instruction_text = 'Think of your target letter \nGet ready'

        update_figure(text, welcome_text)
        print('looking for user input')
        looking_for_input = True
        while looking_for_input:
            # without this pause while true is too fast
            pause(1)
            if start == 1:
                looking_for_input = False
                bufhelp.sendEvent('start', state)
                # debug:
                # print(state)
        sequences = [random_sequence() for _ in range(num_of_cues)]
        # target_show_time = wait_for_trial = interval_stimulus = 1e-4
        # for every cue in the number of cues
        for cue in range(num_of_cues):
            # get sequence
            sequence = sequences[cue]
            # check if the state is test
            # if so present welcome message
            if state == 'test':
                # show the instruction text
                update_figure(text, instruction_text)
                # w ait for x seconds
                pause(target_show_time)
                # clear the figure
                update_figure(text, '')
                # wait for trial to start
                pause(wait_for_trial)

            # run through the letters in the sequence
            for idx, stim in enumerate(sequence):
                # first case is speciarunl:
                if idx == 0:
                    # bufhelp.sendEvent('start', '0')
                    # in calibration light target up green
                    # print(stim)
                    if state == 'calibration':
                        # send to buffer; but different label than normal targetstim
                        # the subject is still focusing on the stimuus now, and
                        # should not be included in the analysis of p300
                        bufhelp.sendEvent('start target', stim)
                        targetstim = stim
                        # show target
                        update_figure(text, stim, target = 'target')
                        # show target time
                        pause(target_show_time)
                        # clear figure
                        update_figure(text, '')
                        # wait time
                        pause(wait_for_trial)
                    # skip in test case
                    elif state == 'testing':
                        # no target stim
                        targetstim == ''
                        continue
                    # this is the real running
                else:
                    update_figure(text, stim)
                    if state == 'calibration' and stim == targetstim:
                        bufhelp.sendEvent('target', stim)
                    else:
                        bufhelp.sendEvent('stimulus', stim)
                    pause(interval_stimulus)

            # show the prediction in test case
            if state == 'test':
                bufhelp.sendEvent('run', 'end')
                # after the runs clear the figure
                update_figure(text, ' ')
                # wait at most target_show_time
                tic = time.time()
                event = bufhelp.waitforevent("classifier.prediction", target_show_time* 1000)
                # in case of timer running out just predict A
                if event == None:
                    pred = 'Classification failed'
                else:
                    pred = event.value
                elap = time.time() - tic
                # print('> elapsed', elap)
                if elap > 1:
                    elap = elap - 1
                else:
                    pause(wait_for_trial - elap)
                    elap = 0
                # wait
                pause(show_prediction)
                bufhelp.sendEvent('prediction', pred)
                update_figure(text, pred, target = 'prediction')
                # print('elapse', elap)
                # normalize the wait time
                pause(target_show_time - elap)
        bufhelp.sendEvent(state, "end")
        pause(1)
        # exit screen
        exit_text = 'Thank you for participating!'
        update_figure(text, exit_text)

    def press(event):
        '''
        looks for key press events
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

     # setting up figure
    fig, ax = subplots(1,1)
    fig.canvas.set_window_title(state)

    # connect with press event registry
    fig.canvas.mpl_connect('key_press_event', press)
    [p] = ax.plot([],[])

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    # text object for plotting
    text = ax.text(\
            0.5*(left+right),\
            0.5*(bottom+top), '',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20, color='white',
            transform=ax.transAxes)
    # black background
    ax.set_facecolor('k')

    # remove xy labels
    ax.set_xticks([])
    ax.set_yticks([])

    # connect to buffer
    import sys, time, bufhelp
    ftc, hdr = bufhelp.connect()

    # ani = test_phase()
    # ani = calibration_phase()
    # run_experiment(0,state, num_of_cues = 1)
    run_experiment(0,state)

    # send message to train classifier to train classifier
    # give time for everything to cool down
    pause(2)
    if state == 'calibration':
        bufhelp.sendEvent('start', 'train')
    elif state == 'test':
        bufhelp.sendEvent('start', 'exit')
    pause(1)
    show(0)
    close(fig)
