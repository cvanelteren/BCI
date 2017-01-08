
import mne
from mne import io
from mne.datasets import sample

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# import a linear classifier from mne.decoding
from mne.decoding import LinearModel

# print(__doc__)
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(2, 15)  # replace baselining with high-pass
events = mne.read_events(event_fname)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    decim=4, baseline=None, preload=True)

labels = epochs.events[:, -1]

# get MEG and EEG data
meg_epochs = epochs.copy().pick_types(meg=True, eeg=False)
meg_data = meg_epochs.get_data().reshape(len(labels), -1)
eeg_epochs = epochs.copy().pick_types(meg=False, eeg=True)
eeg_data = eeg_epochs.get_data().reshape(len(labels), -1)
