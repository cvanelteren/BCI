from pylab import *
import matplotlib.animation as animation
import string
import numpy as np
# matplotlib.use('TkAgg')

# check matplotlib version
# we need to have the latest versions otherwise some modules are not
# loading properly    print(len(sequence[0]))

import sys
sys.path.append('../../')
if not matplotlib.__version__[0] == '2':
    import pip
    pip.main(['install', 'matplotlib==2.0.0b4'])

def experiment(state):
    def random_sequence(n_repeats = 5):
        '''
        returns a random sequence with target letter
        '''
        # define the store vector:
        sequence = [[]] * n_repeats
        # only upper case letters
        letters = string.ascii_uppercase
        alphabet = [x for x in letters]
        for i in range(n_repeats):
            # random shuffle
            # first index is the target
            random_order = np.random.randint(0,len(letters))
            np.random.shuffle(alphabet)
            # tmp = alphabet
            tmp = np.hstack((alphabet[random_order], alphabet))
            sequence[i] = tmp
            # store the letters
            # for j in letters:
            #     sequence[i].append(letters[letters[j]])
        # print(sequence); assert 0
        return sequence


    def update_figure(text, letter, target = 0):
        '''
        Updates the figure with whatever is in letter
        '''
        # update figure with the letter
        text.set_text(letter)
        # target is green
        if target   == 1:
            text.set_color('green')
        # blue for testing phase, shows the prediction
        elif target == 2:
            text.set_color('blue')
        # within the trial we have white letters
        else:
            text.set_color('white')


    def calibration_phase():
        '''
        To do:
            get ready and space bar before starting
        '''
        ani = animation.FuncAnimation(fig, run_experiment, blit = 0, fargs = ('calibration',))
        return ani

    def test_phase():
        # run the experiment
        ani = animation.FuncAnimation(fig, run_experiment, blit = 1, fargs = ('test',))
        return ani

    def run_experiment(i, state, num_of_cues = 5,  target_show_time  = 2, wait_for_trial = 1,\
                        interval_stimulus = .1,\
                        show_prediction = 2
                        ):
        '''
        This runs the basic experiment
        The default parameters set up the amount of loops and the timing between
        stimulus presentation
        '''
        # print(num_of_cues, target_show_time)
        # print ('in the zone')
        # if test phase show welcome message

        # if state == 'test':
        # lookingstopevents for start indicator
        global start
        # welcome the user
        welcome_text = ('Press spacebar to continue')

        instruction_text = 'Think of your target letter \nGet ready'

        update_figure(text, welcome_text)
        looking_for_input = True
        print('looking for user input')
        while looking_for_input:
            # without this pause while true is too fast
            pause(.1)
            if start == 1:
                looking_for_input = False
                bufhelp.sendEvent('start', state)

        # print(state)
        # bufhelp.sendEvent('start', state)
        # pause(.1)
        # bufhelp.sendEvent('')
        # generate the stimuli + targets
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
            for run in sequence:
                # pause(.2)
                # print(run)
                for idx, stim in enumerate(run):
                    # first case is special:
                    if idx == 0:
                        # bufhelp.sendEvent('start', '0')
                        # in calibration light target up green
                        # print(stim)
                        if state == 'calibration':
                            # send to buffer
                            bufhelp.sendEvent('target', stim)
                            targetstim = stim
                            # show target
                            update_figure(text, stim, target = 1)
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
                    x = bufhelp.waitforevent('classifier.prediction', target_show_time)
                    if len(x) > 1:
                        prediction, elapsed_time = x
                    else:
                        elapsed_time = x
                        prediction = 'dunno'

                    # wait
                    pause(show_prediction)
                    # show the prediction in blue
                    # to do: get this from the classifier

                    pred = prediction
                    bufhelp.sendEvent('prediction', pred)
                    update_figure(text, pred, target = 2)
                    pause(target_show_time - elapsed_time)
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
        print('press', event.key)
        # spacebar is registered as ' ' for some reason
        if event.key == ' ':
            start = 1


    # state = 'calibration'
    # state = 'test'
    # close('all')
    global start
    start = 1
     # setting up figure
    fig, ax = subplots(1,1)
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

    import sys
    sys.path.append('../../../python/signalProc/')
    import bufhelp

    ftc, hdr = bufhelp.connect()

    # ani = test_phase()
    # ani = calibration_phase()
    run_experiment(0,state)
    # assert 0


    # send message to train classifier
    if state == 'calibration':
        pause(1)
        bufhelp.sendEvent('start', 'train')
    # show(False)
    show(False)
    # pause(10)
