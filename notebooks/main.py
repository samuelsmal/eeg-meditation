# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Analysis using raw functions (not the MNE package)


# <markdowncell>

# ## imports

# <codecell>

import sys

# adds the library path
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

import pandas as pd
import numpy as np
import seaborn as sns
from functools import reduce, partial
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from functional import seq
from scipy import signal
from itertools import combinations 
import mne

from importlib import reload

from nfme.utils.jupyter import display_video
from nfme.config import config
from nfme.utils import data_loading as data
from nfme.features import bandpower

#%matplotlib inline

# <codecell>

if [int(i) for i in mne.__version__.split('.')] < [0, 20, 0]:
    print('should be at least 0.20.0')
    stop # this is the best hack, it clearly does what it is supposed to do

# <codecell>

from pprint import PrettyPrinter

PrettyPrinter(indent=4).pprint(config)

# <markdowncell>

# ## data loading and bandpower computation

# <codecell>

signal_data_sam = data.load_combined_data(subject='sam')
signal_data_adelie = data.load_combined_data(subject='adelie')

# <codecell>

layout = data.load_layout()

# <codecell>

reload(bandpower)

dflt_values = {
    'bands': config['bands'],
    'window_size': config['bandpower_window_width'],
    'sampling_frequency': config['sampling_frequency'],
} 

bandpowers_sam = bandpower.compute_bandpowers(signal_data_sam, 
                                              subject='sam',
                                              **dflt_values)

bandpowers_adelie = bandpower.compute_bandpowers(signal_data_adelie, 
                                                 subject='adelie',
                                                 **dflt_values)


# <codecell>

reload(bandpower)
bandpowers_sam_30s = bandpower.compute_bandpowers(signal_data_sam, 
                                                  subject='sam',
                                                  **{**dflt_values, 'window_size': '30s'})

bandpowers_adelie_30s = bandpower.compute_bandpowers(signal_data_adelie, 
                                                     subject='adelie',
                                                     **{**dflt_values, 'window_size': '30s'})

# <markdowncell>

# ## audio

# <codecell>

#alpha_over_theta_meditation = bandpowers_sam_30s['meditation'][0]['alpha'] / bandpowers_sam_30s['meditation'][0]['theta']
#alpha_over_theta_baseline   = bandpowers_sam_30s['baseline'][0]['alpha']   / bandpowers_sam_30s['baseline'][0]['theta']

# <codecell>

def _ratios_for_recordings_(recording, numerator, denominator):
    return recording[numerator] / recording[denominator]

_alpha_theta_ratio_fn_ = partial(_ratios_for_recordings_, numerator='alpha', denominator='theta')
    
alpha_theta_ratios = {recording_type: [_alpha_theta_ratio_fn_(rec) for rec in recordings.values()] for recording_type, recordings in bandpowers_sam_30s.items()}

# <markdowncell>

# I just pick one of the frontal electrodes...

# <codecell>

electrode_of_interest = 'Fp1'

# <codecell>

# just cuttong some stuff away at the beginning and end to have a cleaner signal
#electrodes_of_interest = ['Fp1', 'T3', 'O1', 'Cz']
electrodes_of_interest = list(alpha_theta_ratios['baseline'][0].columns)
n_recordings = len(alpha_theta_ratios['baseline'])

fig, axs = plt.subplots(ncols=n_recordings, 
                        nrows=len(electrodes_of_interest),
                        figsize=(5 * len(alpha_theta_ratios['baseline']), len(electrodes_of_interest) * 1.4), 
                        sharey=True)


for recording_type, recording in alpha_theta_ratios.items():
    for recording_idx in range(n_recordings):
        for idx, electrode_of_interest in enumerate(electrodes_of_interest):
            axs[idx][recording_idx].plot(recording[recording_idx].iloc[3000: -3000: 10].loc[:, electrode_of_interest].reset_index(drop=True),
                                         label=recording_type)
            
            axs[idx][recording_idx].set_title(electrode_of_interest)
        
        
                                     
        #_t=.loc[alpha_over_theta_baseline.index[3000:-3000:10], electrode_of_interest]
        #axs[idx].plot(_t.index, _t.values, label="baseline")
        ##sns.lineplot(x=_t.index, y=_t.values, ax=axs[idx], label="baseline")
        #_t=alpha_over_theta_meditation.loc[alpha_over_theta_meditation.index[3000:-3000:10], electrode_of_interest]
        #axs[idx].plot(_t.index, _t.values, label="meditation")
        ##sns.lineplot(x=_t.index, y=_t.values, ax=axs[idx], label="meditation")
        #axs[idx].set_title(electrode_of_interest)

fig.suptitle('alpha / theta')
#axs[-1].legend()
fig.tight_layout()
fig.subplots_adjust(top=0.88)

# <codecell>

_t=alpha_over_theta_meditation.loc[alpha_over_theta_meditation.index[3000:-3000:10], electrode_of_interest]

# <codecell>

_t.index

# <codecell>

# normalising it using the baseline to be between 0 and 1


# <codecell>

alpha_over_theta_meditation.loc[alpha_over_theta_meditation.notna().any(axis=1), :].describe()

# <codecell>

alpha_over_theta_meditation.loc[:, 'Fp1'].describe()

# <markdowncell>

# # Video

# <codecell>

from nfme.utils import video
normalised_pos = video.normalise_layout_pos(layout)

#video_file_names = [gen_topomap_video(bp2['sam']['meditation'][band], normalised_pos, f"{band}-{subject}-{recording_type}-{recording_id}") for b in cfg['bands'].keys()]

# <codecell>

reload(video)
#frames = video.gen_topomap_frames_all_bands(bandpowers_sam_30s['baseline'][0], normalised_pos, fraction_to_plot=0.01)

video.save_frames(frames, 'baseline_all_bands_30s.mp4')

# <codecell>

display_video('baseline_all_bands_30s.mp4')

# <codecell>

stop

# <codecell>

def gen_topomap_video_all(data, normalised_pos, title, fraction_to_plot= 0.01):
    n_plots = np.int(data.shape[0] * fraction_to_plot) # // cfg['sampling_frequency']
    bandpower_over_time_index = data.index
    times_index_to_plot = np.linspace(start=0, stop=bandpower_over_time_index.shape[0] - 1, num=n_plots, dtype=np.int)
    
    frames = gen_topomap_frames(data=data,
                                times_index_to_plot=times_index_to_plot,
                                pos=normalised_pos,
                                title=title)

    movie_file_name = f"{title}.mp4"
    save_frames(frames, movie_file_name)
    return movie_file_name

# <codecell>

stop

# <codecell>

meditation_bandpower = get_bandpower_for_electrode(meditation_pd, electrode=electrode_of_interest, config=cfg)
baseline_bandpower   = get_bandpower_for_electrode(baseline_pd, electrode=electrode_of_interest, config=cfg)

# <codecell>

# for each of the evoked chanels
meditation_bandpower = pd.DataFrame(meditation_bandpower)

# <codecell>

bandpower_adelie = {
    'baseline': get_bandpower_for_electrode(baseline_adelie_pd, electrode=electrode_of_interest, config=cfg),
    'meditation': get_bandpower_for_electrode(meditation_adelie_pd, electrode=electrode_of_interest, config=cfg)
}

# <codecell>

plot_raw_signal(baseline_pd);

# <codecell>

plot_raw_signal(meditation_pd);

# <codecell>

plot_bandpowers(baseline_bandpower, electrode=electrode_of_interest);

# <codecell>

plot_bandpowers(meditation_bandpower, electrode=electrode_of_interest);

# <codecell>

aggregated_power_adelie = aggregate_bandpower(baseline=bandpower_adelie['baseline'], signal=bandpower_adelie['meditation'])
aggregated_power_adelie

# <codecell>

aggregated_power_sam = aggregate_bandpower(baseline=baseline_bandpower, signal=meditation_bandpower)
aggregated_power_sam

# <markdowncell>

# ## spectrogram videos

# <codecell>

fif_meditation = load_raw_mne_from_fif('meditation', subject='sam', config=cfg)

# <codecell>

meditation_csd = mne.preprocessing.compute_current_source_density(fif_meditation)

# <codecell>

meditation_csd.plot(scalings='auto')
meditation_csd.plot_psd()

# <codecell>

def raw_to_epochs(raw, events, sampling_frequency, weird_epoch_offset=100):
    return mne.Epochs(raw=raw, events=events, tmax=events[-1, 0] * 1 / sampling_frequency - weird_epoch_offset).average()

# <codecell>

signals_meditation, events_meditation = load_raw_mne_from_fif('meditation', subject='sam', config=cfg)
signals_baseline, events_baseline = load_raw_mne_from_fif('baseline', subject='sam', config=cfg)

# <codecell>

signals_baseline.plot_sensors(show_names=True)

# <codecell>

epochs_meditation = raw_to_epochs(raw=signals_meditation, events=events_meditation, sampling_frequency=cfg['sampling_frequency'])
epochs_meditation.plot_topomap()

# <codecell>

epochs_baseline = raw_to_epochs(raw=signals_baseline, events=events_baseline, sampling_frequency=cfg['sampling_frequency'])
epochs_baseline.plot_topomap()

# <codecell>

epochs_baseline.info

# <codecell>

def plot_topomap_over_time(title, epochs, events, sampling_frequency, n_plots=64, weird_epoch_offset=100):
    last_frame_in_seconds = np.floor(events[-1, 0] * 1 / sampling_frequency - weird_epoch_offset)
    all_times = np.linspace(0, last_frame_in_seconds, n_plots)
    return epochs.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto', title=title)

# <codecell>

plot_topomap_over_time(epochs=epochs_baseline,
                       events=events_baseline,
                       sampling_frequency=cfg['sampling_frequency'],
                       title='baseline')

# <codecell>

plot_topomap_over_time(epochs=epochs_meditation,
                       events=events_meditation,
                       sampling_frequency=cfg['sampling_frequency'],
                       title='meditation')

# <codecell>

def epochs_to_animation(file_name, epochs, events, sampling_frequency, n_frames, weird_offset=100):
    to_file_parameters = {'show': False, 'blit': False}

    #fig, anim = epochs.animate_topomap(ch_type='eeg', times=np.arange(0, 40 events[-1, 0] * 1 / cfg['sampling_frequency'] - 10, 0.5),  butterfly=True)
    fig, anim = epochs.animate_topomap(ch_type='eeg', 
                                       times=np.linspace(0, events[-1, 0] * 1 / sampling_frequency - weird_offset, n_frames),
                                       butterfly=True, 
                                       **to_file_parameters)
    anim.save(f"{file_name}.mp4")

# <codecell>

epochs_to_animation('baseline', 
                    epochs=epochs_baseline,
                    events=events_baseline,
                    sampling_frequency=cfg['sampling_frequency'],
                    n_frames=10)

# <codecell>

epochs_to_animation('meditation', 
                    epochs=epochs_meditation,
                    events=events_meditation,
                    sampling_frequency=cfg['sampling_frequency'],
                    n_frames=10)

# <codecell>

stop

# <codecell>

reject = dict(eeg=180e-6, eog=150e-6)
event_id, tmin, tmax = {'left/auditory': 1}, -0.2, 0.5
events = mne.read_events(event_fname)
epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                     reject=reject)

evoked_no_ref = mne.Epochs(raw, **epochs_params).average()

title = 'EEG Original reference'
evoked_no_ref.plot(titles=dict(eeg=title), time_unit='s')
evoked_no_ref.plot_topomap(times=[0.1], size=3., title=title, time_unit='s')

# <markdowncell>

# ## 

# <markdowncell>

# ## 

# <markdowncell>

# # Graveyard, not interesting below here

# <codecell>

sampling_rate = 300
window_size = 4 * sampling_rate # in seconds

plt.figure(figsize=(24, 10))
for c in [c for c in signals_pd.columns if c not in ['TRIGGER', 'X1', 'X2', 'X3', 'A2']]:
    freqs, psd = signal.welch(signals_pd.loc[:, c], sampling_rate, nperseg=window_size)

    plt.plot(freqs, psd, label=c)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    #plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    #plt.xlim([0, freqs.max()])
    plt.xlim([0, 20])
    sns.despine()
    
plt.legend();

# <codecell>

# for a window size of... compute the power and compare it over time



bandpower(signals_pd.loc[:, 'T5'], 300., cfg['bands']['theta'])

# <codecell>

signals = baseline['signals']
fig, axs = plt.subplots(nrows=signals.shape[1], figsize=(40, 1.4 * signals.shape[1]))
for channel in range(signals.shape[1]):
    sns.lineplot(data=signals[::10, channel], ax=axs[channel], )
    axs[channel].set_ylabel(baseline['ch_names'][channel])
    
axs[-1].set_xlabel('time [ms]');

# <codecell>

38052/60
