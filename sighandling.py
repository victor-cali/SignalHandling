from scipy.io import loadmat
from numpy import transpose
from numpy import arange
from numpy import zeros
from numpy import ones
from numpy import array
from numpy import int64
import mne

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
