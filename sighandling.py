from scipy.io import loadmat
from numpy import transpose
from numpy import arange
from numpy import zeros
from numpy import ones
from numpy import array
from numpy import int64
import mne

from mne.channels import make_dig_montage
from mne.io import RawArray
from mne import Epochs
from mne import create_info
import numpy as np
import scipy.io

# Auxiliar method to load data from the 
def get_dataset_bbcic4_2b(subject):
    # Load data from .mat file
    dataset = loadmat(subject)
    # Get classes with shape (n_trials, n_channels, n_samples)
    data_c1 = transpose(dataset['C1'],(2,0,1))
    data_c2 = transpose(dataset['C2'],(2,0,1))
    # Shorcuts for functions used to form the Events array
    f0 = lambda class_:zeros(class_.shape[0])
    f1 = lambda class_:ones(class_.shape[0])
    a  = lambda class_:arange(class_.shape[0])
    b  = lambda class_:arange(class_.shape[0])
    # Set standard info for the data
    info = mne.create_info(('C3','Cz','C4'), 250, ('eeg', 'eeg', 'eeg'))
    # Build events array for each class
    events_c1 = array((a(data_c1),f0(data_c1),f1(data_c1)),dtype=int64).transpose()
    events_c2 = array((b(data_c2),f0(data_c2),f0(data_c2)),dtype=int64).transpose()
    # Build epochs for each class
    epoch_c1 = mne.EpochsArray(data_c1,info,events=events_c1,event_id={'left': 1})
    epoch_c2 = mne.EpochsArray(data_c2,info,events=events_c2,event_id={'right':0})
    # Balance classes to same number of trials
    mne.epochs.equalize_epoch_counts((epoch_c1,epoch_c2))
    # Concatenate classes to a single epoch
    epoch = mne.concatenate_epochs((epoch_c1,epoch_c2))
    # Return single epoch with all data
    return epoch

#data = get_dataset2a('D:\Dev\BCIProject\data\dataset2a\S1.mat')

def get_dataset_bbcic3_4a(subject):
    # Load data from .mat file
    dataset = scipy.io.loadmat(subject)
    # DATA
    cnt = dataset['cnt'].transpose()
    stim = np.empty((1, cnt.shape[1]))
    data = np.concatenate((cnt, stim))
    # CUES
    mrk = dataset['mrk']
    y = mrk[0, 0][1]
    y = y[0,~ np.isnan(y[0,:])]
    len_y = len(y)
    ignored = np.zeros(len_y)
    pos = mrk[0, 0][0][0,0:len(y)]
    className = mrk[0, 0][2][0,:]
    event_id = {className[0][0]: 1, className[1][0]: 2}
    # INFO
    info = dataset['nfo']
    name = info['name']
    # Sampling Frequency
    sfreq = info['fs'][0,0][0,0]
    # Channel names
    clab = info['clab'][0, 0]
    len_ch = len(clab[0, :])
    ch_names = [clab[0, i][0] for i in range(len_ch)]
    ch_names.append('STI 014')
    # Channel types
    ch_types = ['eeg' for i in range(len_ch)]
    ch_types.append('stim')
    # Montage positions
    xpos = info['xpos'][0,0]
    xpos = [x for x in xpos[:,0]]
    ypos = info['ypos'][0,0]
    ypos = [y for y in ypos[:,0]]
    ch_pos = {ch: [x,y,0] for ch,x,y in zip(ch_names, xpos, ypos)}
    # Info instance
    info = create_info(ch_names, sfreq, ch_types)
    # Montage instance
    montage = make_dig_montage(ch_pos)
    # Events
    events = np.array((pos,ignored,y), dtype = np.int64).transpose()
    # Raw MNE instance
    raw = RawArray(data, info)
    raw.set_montage(montage)
    raw.add_events(events, stim_channel = 'STI 014')
    # Epochs MNE instance
    tmin, tmax = -0.5, 2.
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks = ('C3', 'Cz', 'C4'), baseline = None, preload=True)
    epochs.equalize_event_counts()
    epochs = epochs.filter(l_freq = 0.5, h_freq = 30.0, method='iir')
    epochs.resample(250.0)
    return epochs