import numpy as np
import string
from pylab import *


def randomSequence(n_repeats = 5):
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

def setupFigure(state):
    '''
    Sets up basic figure window:
        empty, black backgroun with text object
    '''
    # setting up figure
    fig, ax = subplots(1,1)
    fig.canvas.set_window_title(state)


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
    return fig, ax, text

def updateFigure(text, letter, target = 'stimulus'):
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
