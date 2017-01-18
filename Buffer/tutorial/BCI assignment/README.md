# INSTRUCTIONS - lazy start - Run startScript.sh on linux: note this assumes
- run startScript.sh on linux (assumed gnome-terminal); this was tested and
works
- Run startScript.bat on windows; note this wasn't tested; my machine runs
linux




# INSTRUCTIONS - manual
- start bufferInterface/SignalProcessing.py
- start debug_eeg - use the experiment function :
 - experiment('calibration')
 - experiment('test')
*Note* one can call the this script from the command line by: python -c "from
experiment import experiment as exp; exp('state')", where state is either
calibration or test.

The figures will automatically close and open. The title indicates the state of
the experiment you are in (calibration / test).

The experiment('calibration') function automatically trains the classifier and stores
 the data

*Note*: this package assumes that it is within the tutorial directory.
That is, Buffer assignment is in the tutorial directory.

# GENERAL IDEA
The idea was to have one analysis script, that when running the experiment is
the only script you would want to edit, i.e. if you want a different classifier
one would just need to edit one line of code in one script. The experiment is
already set. Furthermore, as this is an exercise a lazy start would be nice,
i.e. a script that starts the buffer, starts the signal processing, and runs tje
experiment. Offcourse in the real world one might want to run the experiment,
signal processing, and buffer on separate machines, and this is still possible!

The data is stored in the Data subdirectory, and the figures will be stored in the Figures subdirectory. Currently, only the ERPs of the target and the non-targets are saved.

# SCRIPTS OVERVIEW
- Experiment.py ; controls the experiment in both test and calibration phase. Also
send event to train the classifier after calibration.
- controlScript.py ; for some reason you can't call subprocess.call in another
directory and hence I made a control script that is in the same directory as
SignalProcessing.py such that it remains active in the terminal. Note this is
not an issue when running  manual, but it is for some reason when running in
lazy start.
- SignalProcessing.py : can be viewed as a controller script as well. This
script interfaces with the buffer, i.e. initiates calibration data collection,
initiates training of classifier, initiates testing phase.
- preproc.py : I wrote some basic functions that does preprocessing, i.e
detrending, formatting data for a binary classifier, filtering etc. *Note* this
is all a work in process. I believe the idea of this assignment was to make the
framework, and this is exactly what I did. The preprocessing script can be
extended trivially from this point.
- classficiation.py : wrapper for functions from sklearn; I used a logistic
regressor for the classification. Again, this can be easily extended to include
other classifiers.
- FeedbackSignals.py : in my opinion this is a redundant script, it takes input from the buffer
in the feedback phase and produces the prediction from the classifier. 

#PACKAGES
- Matplotbib 2.0b4 : this is needed as otherwise some functions in the
experiment won't work
- numpy   : linear algebra
- scipy   : sciency things
- sklearn : sciency things
- seaborn : plotting
