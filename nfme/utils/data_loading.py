#    NFME NeuroFeedback for Meditation using EEG recordings
#    Copyright (C) 2020 Samuel von Baussnern, samuel.edlervonbaussnern@epfl.ch
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import mne
import pandas as pd
from pathlib import Path

from nfme.features import preprocessing
from nfme.config import config


def load_signal_data(data_type, subject='sam', recording=0, apply_signal_cropping=True,
                     apply_bandpass_filter=True, apply_remove_reference_signal=True):
    """loads the data and returns a pandas dataframe

    Parameters
    ----------
    data_type : string
      type of the data, right now two options are valid: `baseline` or `meditation`
    subject: string
      name of the subject
    recording: int
      number of recording, if you have multiple of same type and subject

    Returns
    -------
    a pandas dataframe, timedeltaindexed of the raw signals
    """
    subject_path_config = config.value('paths', 'subjects', subject)
    p = Path(f"{config.value('paths', 'base')}/{subject_path_config['prefix']}/offline/"
                           f"{subject_path_config['recordings'][data_type][recording]}"
                           "-raw.pcl").absolute()
    data = pd.read_pickle(p)

    _t = data['timestamps'].reshape(-1)
    _t -= _t[0]

    index = pd.TimedeltaIndex(_t, unit='s')
    data = pd.DataFrame(data=data['signals'],
                        columns=data['ch_names'])\
               .drop(columns=config['columns_to_remove'])


    if apply_signal_cropping:
        data = data.iloc[config['default_signal_crop'], :]
        index = index[config['default_signal_crop']]

    if apply_bandpass_filter:
        data = data.apply(lambda xs: preprocessing.bandpass_filter(xs,
                                                                   config['bandpass_filter'],
                                                                   config['sampling_frequency']),
                          axis=0)

    if apply_remove_reference_signal:
        data = preprocessing.remove_reference_signal(data, config['reference_electrode'])

    data.index = index

    return data


def get_channelsList(subject='adelie'):
    subject_paths = config.value('paths', 'subjects', subject)
    base_path = config.value('paths', 'base')
    file_path = f"{base_path}/{subject_paths['prefix']}/offline/fif/{config.value('paths', 'channels_file')}"

    with open(file_path, 'r') as channels_file:
        all_channels = channels_file.read().strip()

    return [channel for channel in all_channels.split('\n') if channel not in config['columns_to_remove']]


def load_raw_mne_from_fif(data_type, subject='adelie', recording=0, montage='standard_1020'):
    """loads the data and returns an instance of mne.Raw

    Parameters
    ----------
    data_type : string
      type of the data, right now two options are valid: `baseline` or `meditation`
    subject: string
      name of the subject
    recording: int
      number of recording, if you have multiple of same type and subject
    montage: string
      the type of montage that was used for the recording see:
      https://mne.tools/dev/generated/mne.channels.make_standard_montage.html

    Returns
    -------
    a mne.Raw instance that has the correct montage and info and is ready to be plotted
    """
    subject_paths = config.value('paths', 'subjects', subject)
    base_path = config.value('paths', 'base')
    recording_id = subject_paths['recordings'][data_type][recording]
    file_path = f"{base_path}{subject_paths['prefix']}/offline/fif/{recording_id}-raw.fif"

    # Create a digitization of the montage
    digitization = mne.channels.make_standard_montage(montage)
    channels = get_channelsList(subject=subject)

    # Read from fif file
    raw = mne.io.read_raw_fif(file_path, preload=True)

    # Create info with some useful information
    info = mne.create_info(channels, sfreq=config['sampling_frequency'], ch_types='eeg')
    raw.info = info

    # set the montage
    raw.set_montage(digitization)

    raw = raw.pick_types(eeg=True, stim=False)
    raw.set_eeg_reference(projection=True).apply_proj()

    return raw


def load_raw_mne_from_fif(data_type, subject='adelie', recording=0, montage='standard_1020'):
    """loads the data and returns an instance of mne.Raw

    Parameters
    ----------
    data_type : string
      type of the data, right now two options are valid: `baseline` or `meditation`
    subject: string
      name of the subject
    recording: int
      number of recording, if you have multiple of same type and subject
    montage: string
      the type of montage that was used for the recording see: https://mne.tools/dev/generated/mne.channels.make_standard_montage.html

    Returns
    -------
    a mne.Raw instance that has the correct montage and info and is ready to be plotted
    """
    subject_paths = config.value('paths', 'subjects', subject)
    base_path = config.value('paths', 'base')
    recording_id = subject_paths['recordings'][data_type][recording]
    file_path = f"{base_path}{subject_paths['prefix']}/offline/fif/{recording_id}-raw.fif"

    # Create a digitization of the montage
    digitization = mne.channels.make_standard_montage(montage)
    channels = get_channelsList(subject=subject)

    # Read from fif file
    raw = mne.io.read_raw_fif(file_path, preload=True)

    # I hope that this is correct
    raw, _ = mne.set_eeg_reference(raw, [config['reference_electrode']])

    # finding events
    events = mne.find_events(raw, stim_channel='TRIGGER')

    # Create info with some useful information
    raw.info = mne.create_info(channels, sfreq=config['sampling_frequency'], ch_types='eeg')
    # set the montage
    raw.set_montage(digitization)

    raw = raw.pick_types(eeg=True, stim=False)
    raw.set_eeg_reference(projection=True).apply_proj()

    return raw, events


def load_combined_data(subject, apply_remove_reference_signal=True, apply_bandpass_filter=True,
                       apply_signal_cropping=True):
    """loads the all the data for the given subject
    """

    dftl_args = {
        'subject': subject,
        'apply_signal_cropping': apply_signal_cropping,
        'apply_bandpass_filter': apply_bandpass_filter,
        'apply_remove_reference_signal': apply_remove_reference_signal
    }

    data = {recording_type: [load_signal_data(**dftl_args, data_type=recording_type, recording=rec)
                             for rec in range(len(recordings))]
            for recording_type, recordings in config.value('paths', 'subjects', subject, 'recordings').items()}

    return data

def load_layout():
    signals, _= load_raw_mne_from_fif('baseline', subject='sam')
    return mne.channels.make_eeg_layout(signals.info)
