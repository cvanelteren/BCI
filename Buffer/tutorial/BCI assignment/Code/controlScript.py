from subprocess import call
import sys, os

# sys.path.append('bufferInterface/')
call(['gnome-terminal', '-x', 'python', 'SignalProcessing.py'])
call(['gnome-terminal', '-x', '../../../debug_quickstart.sh'])
# call(['gnome-terminal', '-x', '../../../debug_quickstart.sh'])


sys.path.append('../')
# sys.path.append('../')
from experiment import experiment as e
e('calibration')
e('test')
