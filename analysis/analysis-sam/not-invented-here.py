# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Analysis using raw functions (not the MNE package)


# <markdowncell>

# ## imports

# <codecell>

import pandas as pd
import numpy as np
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from functional import seq
from scipy import signal
from itertools import combinations 
import mne

#%matplotlib qt 

# <codecell>

if [int(i) for i in mne.__version__.split('.')] < [0, 20, 0]:
    print('should be at least 0.20.0')
    stop # this is the best hack, it clearly does what it is supposed to do

# <markdowncell>

# ## helper functions

# <markdowncell>

# ### computation

# <codecell>

def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.
    Source: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper
    

    band = np.asarray(band)
    low, high = band
    
    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        if data.shape[0] < nperseg:
            return np.NaN
            
        
        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
        
    return bp

# <codecell>

def get_bandpower_for_electrode(signal_data, electrode, bands, sampling_frequenc=300, window_size='1s'):
    """Calculates the bandpower with a rolling window for the given electrode
    
    Note that this will take some time... I suggest that you only use a part of the signal to try it out.
    
    Parameters
    ----------
    signal_data: 2d pandas dataframe
        raw signal data, indexed by a timedeltaindex (or any other time-based index)
    electrode: string
        name of the electrode of interest
    config: dict
        dict of config parameters
    window_size: string
        size of rolling window
        
    Returns
    -------
    a new pandas dataframe of the bandpowers, in addition all ration combinations are listed as well
    """
    bandpowers = {}

    for band_name, band_range in bands.items():
        bandpowers[band_name] = signal_data.loc[:, electrode]\
            .rolling(window_size)\
            .apply(lambda xs: bandpower(xs, sampling_frequency, band_range))

    # compute all different ratios
    for bn_l, bn_r in combinations(bands.keys(), 2):
        bandpowers[f"{bn_l} / {bn_r}"] = bandpowers[bn_l] / bandpowers[bn_r]
        
    return bandpowers 

# <codecell>

def aggregate_bandpower(baseline, signal):
    aggregated_fns = ['mean', 'median', 'min', 'max']
    aggregated_power = pd.DataFrame(index=pd.MultiIndex.from_product([list(baseline.keys()), ['baseline', 'meditation']]),
                                    columns=aggregated_fns)

    for band, power in baseline.items():
        aggregated_power.loc[(band, "baseline"), :] = power.agg(aggregated_fns)


    for band, power in signal.items():
        aggregated_power.loc[(band, 'meditation'), :] = power.agg(aggregated_fns)


    return aggregated_power

# <markdowncell>

# ### data loading

# <codecell>

def load_signal_data(data_type, config, subject='sam', recording=0, remove_references=True):
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
    subject_paths = get_config_value(config, 'paths', 'subjects', subject)
    data = pd.read_pickle(f"{config['paths']['base']}/{subject_paths['prefix']}/offline/{get_config_value(subject_paths, 'recordings', data_type)[recording]}-raw.pcl")
    
    _t = data['timestamps'].reshape(-1)
    _t -= _t[0]

    signal = pd.DataFrame(data=data['signals'], 
                          index=pd.TimedeltaIndex(_t, unit='s'), 
                          columns=data['ch_names'])\
               .drop(columns=config['columns_to_remove'])
    
    return signal.loc[signal.index[config['default_signal_crop']], :]

# <codecell>

def get_channelsList(config, subject='adelie'):
    subject_paths = get_config_value(config, 'paths', 'subjects', subject)
    base_path = get_config_value(config, 'paths', 'base')
    file_path = f"{base_path}/{subject_paths['prefix']}/offline/fif/{get_config_value(config, 'paths', 'channels_file')}"
    
    with open(file_path, 'r') as channels_file:
        all_channels = channels_file.read().strip()
        
    return [channel for channel in all_channels.split('\n') if channel not in config['columns_to_remove']]

def load_raw_mne_from_fif(data_type, config, subject='adelie', recording=0, montage='standard_1020'):
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
    subject_paths = get_config_value(config, 'paths', 'subjects', subject)
    base_path = get_config_value(config, 'paths', 'base')
    recording_id = get_config_value(subject_paths, 'recordings', data_type)[recording]
    file_path = f"{base_path}{subject_paths['prefix']}/offline/fif/{recording_id}-raw.fif"
    
    # Create a digitization of the montage
    digitization = mne.channels.make_standard_montage(montage)
    channels = get_channelsList(config, subject=subject)
    
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

# <codecell>

def load_raw_mne_from_fif(data_type, config, subject='adelie', recording=0, montage='standard_1020'):
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
    subject_paths = get_config_value(config, 'paths', 'subjects', subject)
    base_path = get_config_value(config, 'paths', 'base')
    recording_id = get_config_value(subject_paths, 'recordings', data_type)[recording]
    file_path = f"{base_path}{subject_paths['prefix']}/offline/fif/{recording_id}-raw.fif"
    
    # Create a digitization of the montage
    digitization = mne.channels.make_standard_montage(montage)
    channels = get_channelsList(config, subject=subject)
    
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

# <markdowncell>

# ### plots

# <codecell>

def plot_bandpowers(bandpowers, electrode):
    fig, axs = plt.subplots(nrows=len(bandpowers), sharex=True, figsize=(25, 15))
    time_index = list(bandpowers.values())[0].index
    time_index_as_seconds = [t.total_seconds() for t in time_index]

    for i, (bn, bp) in enumerate(bandpowers.items()):
        axs[i].plot(bp.reset_index(drop=True))
        axs[i].set_ylabel(bn)

    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: index_to_time(x, time_index)))
    fig.suptitle(f"Bandpower of {electrode}")
    
    return fig


def plot_raw_signal(signal_pd, sampling=10):
    """
    Parameters
    ----------
    signal_pd: 2d pandas dataframe
        long-format (a column for each electrode)
    sampling: int
        step size of data points used for plotting
    
    Returns
    -------
    a figure of the plot
    
    """
    
    fig, axs = plt.subplots(nrows=signal_pd.shape[1], figsize=(40, 1.4 * signal_pd.shape[1]), sharex=True)
    for channel_id, channel in enumerate(signal_pd.columns):
        d = signal_pd.loc[::sampling, channel]
        sns.lineplot(data=d.reset_index(drop=True), ax=axs[channel_id])
        axs[channel_id].set_ylabel(channel)

    axs[-1].set_xlabel('time [ms]')
    axs[-1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: index_to_time(x, d.index)))
    
    fig.suptitle("Raw signal recording")
    
    return fig

# <codecell>

def index_to_time(x, time_index, step_size=1):
    """Helper function to add the axis labels"""
    if (x < 0 or x * step_size >= len(time_index)):
        return ''
    
    seconds = time_index[int(x*step_size)].total_seconds()
    return f"{int(seconds/60)}\' {seconds/60:.2f}\""

# <codecell>

def get_config_value(config, *args):
    """Helper to get read the config"""
    return reduce(lambda cfg, val: cfg[val], args, config)

# <markdowncell>

# # main stuff

# <markdowncell>

# ## config

# <codecell>

# This is the config. Add any value that you deem necessary. This should contain everything that belongs to the setup, the filtering pipeline, etc.

cfg = {
    'paths': {
        'base': '../../data/AlphaTheta',
        'channels_file': 'channelsList.txt',
        'subjects': {
            'sam': {
                'prefix': '/sam-AlphaTheta',
                'recordings': {
                    'baseline': [
                        '20200304-144100',
                        '20200304-144601'
                    ],
                    'meditation': [
                         '20200304-144933'   
                    ]
                }
            },
            'adelie': {
                'prefix': '/adelie-AlphaTheta',
                'recordings': {
                    'baseline': [
                        '20200304-151358',
                    ],
                    'meditation': [
                        '20200304-152058',
                    ]
                }
            }
        }
    },
    'columns_to_remove': [
        'TRIGGER', 'X1', 'X2', 'X3',
    ],
    'reference_electrode': 'A2',
    'default_signal_crop': np.s_[3000:-3000], # this corresponds to ~1 second at the beginning and end, given by the sampling frequency
    'sampling_frequency': 300,
    'bands': {
        'gamma': [40, 100],
        'beta':  [12, 40],
        'alpha': [8, 12],
        'theta': [4, 8],
        'delta': [0.5, 4]
    }
}

# <markdowncell>

# ## raw file

# <codecell>

meditation_pd = load_signal_data('meditation', config=cfg)
baseline_pd = load_signal_data('baseline', config=cfg)

# <codecell>

signals, _= load_raw_mne_from_fif('baseline', subject='sam', config=cfg)
layout = mne.channels.make_eeg_layout(signals.info)

# <codecell>

#baseline_adelie_pd = load_signal_data('baseline', subject='adelie', config=cfg)
#meditation_adelie_pd = load_signal_data('meditation', subject='adelie', config=cfg)

# <codecell>

from pathlib import Path
from functools import partial
from multiprocessing import Pool

def bandpower_with_rolling_window(electrode, signal_data, window_size, sampling_frequency, band_range):
        return (electrode, 
                signal_data.loc[:, electrode]\
                           .rolling(window_size)\
                           .apply(lambda xs: bandpower(xs, sampling_frequency, band_range)))
    
recording_type = 'meditation'
recording_id = 0
subject = 'sam'

def load_or_compute(fn, fn_kwargs, file_path, force=False):
    if Path(file_path).is_file() and not force:
        return pd.read_pickle(file_path)
    else:
        ret = fn(**fn_kwargs)
        
        ret.to_pickle(file_path)
        
        return ret
    
fn_params = {
    'window_size': '1s',
    'sampling_frequency': cfg['sampling_frequency'],
    'signal_data': meditation_pd, 
}

bands = list(cfg['bands'].keys())
bands = ['theta']

with Pool() as p:
    for band in bands:
        file_path = f"{band}_over_time_{recording_type}_{recording_id}_{subject}.pkl"
        if Path(file_path).is_file():
            bandpower_over_time = pd.read_pickle(file_path)
        else:
            fn = partial(bandpower_with_rolling_window, **{**fn_params, 'band_range': cfg['bands'][band]})
            bandpower_over_time =  pd.DataFrame(dict(p.map(fn, list(meditation_pd.columns))))
            bandpower_over_time.to_pickle(file_path)
            
        

# <codecell>

from mne import viz
import matplotlib.animation as animation

_t = layout.pos[:, :2]
_t -= np.mean(_t, 0)
_t /= np.max(_t, 0)

n_plots = 64

bandpower_over_time_index = bandpower_over_time.index

times_index_to_plot = np.linspace(start=0, stop=bandpower_over_time_index.shape[0] - 1, num=n_plots, dtype=np.int)


#fig = plt.figure(figsize=(20, 20))
#ims = []
#for idx, time_index in enumerate(times_index_to_plot):
#    viz.plot_topomap(bandpower_over_time.iloc[time_index, :].T.values, 
#                     sphere=1.,
#                     pos=_t,
#                     sensors=False,
#                     show_names=True,
#                     names=bandpower_over_time.columns)
#    
#
#    plt.title(index_to_time(times_index_to_plot[idx], bandpower_over_time_index))
#    fig.canvas.draw()
#    fig_val = np.array(fig.canvas.renderer._renderer)[:, :, :3]
#    ims.append(fig_val)
#    #plt.close()

# <codecell>

import video
from importlib import reload
reload(video)

# <codecell>

video.create_video(data=bandpower_over_time, times_index_to_plot=times_index_to_plot, pos=_t, file_name='movie.mp4')

# <codecell>

stop

# <codecell>

frames = [] # for storing the generated images
fig = plt.figure(figsize=(20, 20))
for img in ims:
    frames.append(plt.imshow(img, animated=True))

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
#ani.save('movie.mp4')
#plt.close()
plt.show()

# <codecell>

stop

# <codecell>



# <codecell>

plt.imshow(fig_val)

# <codecell>

plt.imshow(ims[0])

# <codecell>

fig = plt.figure(figsize=(20, 20))
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# ani.save('dynamic_images.mp4')
plt.show() 

# <codecell>

fig.suptitle(index_to_time())

# <codecell>

import matplotlib.cm as cm

img = [] # some array of images
frames = [] # for storing the generated images
fig = plt.figure()
for i in xrange(6):
    frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

ani = animation.ArtistAnimation(fig, img, interval=50, blit=True,
                                repeat_delay=1000)
# ani.save('movie.mp4')
plt.show()

# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>

import warnings
from itertools import groupby
import pathlib
import logging
import numpy as np
import cv2
import imageio
import colorsys
import seaborn as sns

from PIL import Image


def plot_drosophila_2d(pts=None, draw_joints=None, img=None, colors=None, thickness=None,
                       draw_limbs=None, circle_color=None):
    """
    taken from https://github.com/NeLy-EPFL/drosoph3D/blob/master/GUI/plot_util.py
    """
    if colors is None:
        colors = skeleton.colors
    if thickness is None:
        thickness = [2] * 10
    if draw_joints is None:
        draw_joints = np.arange(skeleton.num_joints)
    if draw_limbs is None:
        draw_limbs = np.arange(skeleton.num_limbs)
    for joint_id in range(pts.shape[0]):
        limb_id = skeleton.get_limb_id(joint_id)
        if (pts[joint_id, 0] == 0 and pts[joint_id, 1] == 0) or limb_id not in draw_limbs or joint_id not in draw_joints:
            continue

        color = colors[limb_id]
        r = 5 if joint_id != skeleton.num_joints - 1 and joint_id != ((skeleton.num_joints // 2) - 1) else 8
        cv2.circle(img, (pts[joint_id, 0], pts[joint_id, 1]), r, color, -1)

        # TODO replace this with skeleton.bones
        if (not skeleton.is_tarsus_tip(joint_id)) and (not skeleton.is_antenna(
                joint_id)) and (joint_id != skeleton.num_joints - 1) and (
                joint_id != (skeleton.num_joints // 2 - 1)) and (not (
                pts[joint_id + 1, 0] == 0 and pts[joint_id + 1, 1] == 0)):
            cv2.line(img, (pts[joint_id][0], pts[joint_id][1]), (pts[joint_id + 1][0], pts[joint_id + 1][1]),
                     color=color,
                     thickness=thickness[limb_id])

    if circle_color is not None:
        img = cv2.circle(img=img, center=(img.shape[1]-20, 20), radius=10, color=circle_color, thickness=-1)

    return img


#def _get_and_check_file_path_(args, template=SetupConfig.value('video_root_path')):
#    gif_file_path = template.format(begin_frame=args[0], end_frame=args[-1])
#    pathlib.Path(gif_file_path).parent.mkdir(parents=True, exist_ok=True)
#
#    return gif_file_path


def _save_frames_(file_path, frames, format='mp4', **kwargs):
    """
    If format==GIF then fps has to be None, duration should be ~10/60
    If format==mp4 then duration has to be None, fps should be TODO
    """
    if format.lower() == 'gif':
        _kwargs = {'duration': 10/60}
    elif format.lower() == 'mp4':
        _kwargs = {'fps': 24}

    pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16')
        imageio.mimsave(file_path, frames, format=format, **{**_kwargs, **kwargs})


def _add_frame_and_embedding_id_(frame, emb_id=None, frame_id=None, color=None):
    params = {"org": (0, frame.shape[0] // 2),
              "fontFace": 1,
              "fontScale": 1,
              "color": color,
              "thickness": 1}

    if emb_id is not None:
       frame = cv2.putText(img=np.copy(frame), text=f"cluster_id: {emb_id:0>3}", **params)

    if frame_id is not None:
       frame = cv2.putText(img=np.copy(frame), text=f"frame_id: {frame_id:0>4}", **{**params, 'org': (params['org'][0], params['org'][1] + 24)})

    return frame


def _float_to_int_color_(colors):
    return (np.array(colors) * 255).astype(np.int).tolist()

def comparision_video_of_reconstruction(positional_data, cluster_assignments, image_id_with_exp, labels,
                                        n_train_data_points, images_paths, cluster_colors=None,
                                        run_desc=None, epochs=None):
    """Creates a video (saved as a gif) with the embedding overlay, displayed as an int.

    Args:
        xs: [<pos data>] list of pos data, of shape: [frames, limb, dimensions] (can be just one, but in an array)
            will plot all of them, the colors get lighter
        embeddings: [<embeddings_id>]
            assumed to be in sequence with `get_frame_path` function.
            length of embeddings -> number of frames
        file_path: <str>, default: SEQUENCE_GIF_PATH
            file path used to get
    Returns:
        <str>                            the file path under which the gif was saved
    """
    text_default_args = {
        "fontFace": 1,
        "fontScale": 1,
        "thickness": 1,
    }


    n_frames = len(images_paths)
    image_height, image_width, _ = cv2.imread(images_paths[0]).shape
    lines_pos = ((np.array(range(n_frames)) / n_frames) * image_width).astype(np.int).tolist()

    _train_test_split_marker = np.int(n_train_data_points / n_frames * image_width)
    _train_test_split_marker_colours = [(255, 0, 0), (0, 255, 0)]

    _colors_for_pos_data = [lighten_int_colors(skeleton.colors, amount=v) for v in np.linspace(1, 0.3, len(positional_data))]

    def pipeline(frame_id, frame):
        f = _add_frame_and_embedding_id_(frame, cluster_assignments[frame_id], frame_id,
                                         color=cluster_colors[cluster_assignments[frame_id]])

        # xs are the multiple positional data to plot
        for x_i, x in enumerate(positional_data):
            f = plot_drosophila_2d(x[frame_id].astype(np.int), img=f, colors=_colors_for_pos_data[x_i])


        # train test split marker
        if n_train_data_points == frame_id:
            cv2.line(f, (_train_test_split_marker, image_height - 20), (_train_test_split_marker, image_height - 40), (255, 255, 255), 1)
        else:
            cv2.line(f, (_train_test_split_marker, image_height - 10), (_train_test_split_marker, image_height - 40), (255, 255, 255), 1)



        # train / test text
        f = cv2.putText(**text_default_args,
                        img=f,
                        text='train' if frame_id < n_train_data_points else 'test',
                        org=(_train_test_split_marker, image_height - 40),
                        color=_train_test_split_marker_colours[0 if frame_id < n_train_data_points else 1])

        # experiment id
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=data.experiment_key(obj=image_id_with_exp[frame_id][1]),
                        org=(0, 20),
                        color=(255, 255, 255))

        # image id
        #_text_size, _ = cv2.getTextSize(**text_default_args, text=experiment_key(obj=image_id_with_exp[frame_id][1]))
        #f = cv2.putText(**text_default_args,
        #                img=f,
        #                text=image_id_with_exp[frame_id][0],
        #                org=(_text_size[0], 20),
        #                color=(255, 255, 255))

        # class label
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=labels[frame_id],
                        org=(0, 40),
                        color=(255, 255, 255))

        f = cv2.putText(**text_default_args,
                        img=f,
                        text=run_desc,
                        org=(0, 60),
                        color=(255, 255, 255))

        # cluster assignment bar
        for line_idx, l in enumerate(lines_pos):
            if line_idx == frame_id:
                cv2.line(f, (l, image_height), (l, image_height - 20), cluster_colors[cluster_assignments[line_idx]], 2)
            else:
                cv2.line(f, (l, image_height), (l, image_height - 10), cluster_colors[cluster_assignments[line_idx]], 1)


        return f

    frames = (pipeline(frame_id, cv2.imread(path)) for frame_id, path in enumerate(images_paths) if is_file(path))

    output_path = f"{SetupConfig.value('video_root_path')}/{run_desc}_e-{epochs}_hubert_full.mp4"
    _save_frames_(output_path, frames, format='mp4')

    return output_path



def plot_embedding_assignment(x_id_of_interest, X_embedded, label_assignments):
    seen_labels = label_assignments['label'].unique()
    _cs = sns.color_palette(n_colors=len(seen_labels))

    fig = plt.figure(figsize=(10, 10))
    behaviour_colours = dict(zip(seen_labels, _cs))

    for l, c in behaviour_colours.items():
        _d = X_embedded[label_assignments['label'] == l]
        # c=[c] since matplotlib asks for it
        plt.scatter(_d[:, 0], _d[:,1], c=[c], label=l.name, marker='.')

    #print(x_id_of_interest)
    _t = label_assignments.iloc[x_id_of_interest]['label']
    #print(_t)
    cur_color = behaviour_colours[_t]
    plt.scatter(X_embedded[x_id_of_interest, 0], X_embedded[x_id_of_interest, 1], c=[cur_color], linewidth=10, edgecolors=[[0, 0, 1]])
    plt.legend()
    plt.title('simple t-SNE on latent space')

    # TODO I would like to move the lower part to a different function, not tested if that works
    # though
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    #fig.canvas.draw_idle()
#
    ## Now we can save it to a numpy array.
    #plot_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)\
    #              .reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    #return plot_data

    fig_val = np.array(fig.canvas.renderer._renderer)[:, :, :3]
    plt.close()
    return fig_val

def combine_images_h(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2, img1.shape[2]), np.uint8)
    vis[:h1, :w1, :] = img1
    vis[:h2, w1:w1+w2, :] = img2
    #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    return vis
    #cv2.imshow("test", vis)



def video_angle(cluster_assignments, images_paths_for_experiments, cluster_id_to_visualize=None, cluster_colors=None, run_desc=None, as_frames=False):
    """
    run_desc refers to the model experimnt id, not fly-experiment


    ... in general stuff in here sucks... big time...
    """
    if cluster_id_to_visualize is None:
        cluster_assignment_idx = list(range(len(cluster_assignments)))
    else:
        cluster_assignment_idx = np.where(cluster_assignments == cluster_id_to_visualize)[0]

    text_default_args = {
        "fontFace": 1,
        "fontScale": 1,
        "thickness": 1,
    }

    cluster_ids = np.unique(cluster_assignments)
    if cluster_colors is None:
        cluster_colors = dict(zip(cluster_ids, _float_to_int_color_(sns.color_palette(palette='bright', n_colors=len(cluster_ids)))))

    image_height, image_width, _ = cv2.imread(images_paths_for_experiments[0][1]).shape
    lines_pos = ((np.array(range(len(cluster_assignments))) / len(cluster_assignments)) * image_width).astype(np.int)[cluster_assignment_idx].tolist()

    def pipeline(frame_nb, frame, frame_id, embedding_id, experiment, experiment_path=None):
        # frame_nb is the number of the frame shown, continuous
        # frame_id is the id of the order of the frame,
        # e.g. frame_nb: [0, 1, 2, 3], frame_id: [123, 222, 333, 401]
        # kinda ugly... note that some variables are from the upper "frame"
        #f = _add_frame_and_embedding_id_(frame, embedding_id, frame_id)
        f = frame

        # experiment id
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=data._key_(experiment),
                        org=(0, 20),
                        color=(255, 255, 255))

        # image id
        _text_size, _ = cv2.getTextSize(**text_default_args, text=data._key_(experiment))
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=pathlib.Path(experiment_path).stem,
                        org=(_text_size[0], 20),
                        color=(255, 255, 255))

        # model experiment description
        f = cv2.putText(**text_default_args,
                        img=f,
                        text=run_desc,
                        org=(0, 40),
                        color=(255, 255, 255))

        # cluster assignment bar
        for line_idx, l in enumerate(lines_pos):
            if line_idx == frame_nb:
                cv2.line(f, (l, image_height), (l, image_height - 20), cluster_colors[cluster_assignments[cluster_assignment_idx[line_idx]]], 2)
            else:
                cv2.line(f, (l, image_height), (l, image_height - 10), cluster_colors[cluster_assignments[cluster_assignment_idx[line_idx]]], 1)


        return f

    frames = (pipeline(frame_nb, cv2.imread(experiment[1]), frame_id, cluster_assignment,
                       experiment[0], experiment_path=experiment[1])
              for frame_nb, (frame_id, cluster_assignment, experiment) in enumerate(zip(
                  cluster_assignment_idx,
                  cluster_assignments[cluster_assignment_idx],
                  np.array(images_paths_for_experiments)[cluster_assignment_idx]))
              if pathlib.Path(experiment[1]).is_file())

    if as_frames:
        return frames
    else:
        output_path = config.EXPERIMENT_VIDEO_PATH.format(experiment_id=run_desc, vid_id=cluster_id_to_visualize or 'all')
        _save_frames_(output_path, frames, format='mp4')

        return output_path

# new video helpers

def _path_for_image_(image_id, label):
    base_path = SetupConfig.value('experiment_root_path')
    exp_path = SetupConfig.value('experiment_path_template').format(base_path=base_path,
                                                         study_id=label.study_id,
                                                         fly_id=label.fly_id,
                                                         experiment_id=label.experiment_id)
    return SetupConfig.value('fly_image_template').format(base_experiment_path=exp_path, image_id=image_id)

def resize_image(img, new_width=304):
    wpercent = (new_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((new_width, hsize), Image.ANTIALIAS)

def pad_with_last(list_of_lists):
    max_len = max([len(i) for i in list_of_lists])

    def _pad_with_last_(ls, to_len):
        diff_len = to_len - len(ls)
        return ls + [ls[-1]] * diff_len

    return [_pad_with_last_(ls, max_len) for ls in list_of_lists]

def group_video_of_cluster(cluster_id, paths, run_desc, epochs, n_sequences_to_draw=9,
                           pad_videos=False):
    images = [[resize_image(Image.open(p)) for p in ax1] for ax1 in paths[:n_sequences_to_draw]]

    if pad_videos:
        images = pad_with_last(images)

    img = images[0][0]

    element_width, element_height = img.size
    n_elements_x_dim = np.int(np.sqrt(n_sequences_to_draw))
    n_elements_y_dim = np.int(np.sqrt(n_sequences_to_draw))

    combined_images = [Image.new('RGB', (n_elements_x_dim * element_width, n_elements_y_dim * element_height)) for _ in range(len(images[0]))]

    for sequence_id, sequence in enumerate(images):
        x_offset = (sequence_id % n_elements_x_dim) * element_width
        y_offset = (sequence_id // n_elements_x_dim) * element_height

        for frame_number, image in enumerate(sequence):
            combined_images[frame_number].paste(image, (x_offset, y_offset))

    #return combined_images, images

    file_path = (f"{SetupConfig.value('video_root_path')}"
                 f"/group_of_cluster-{cluster_id}-{run_desc}-e-{epochs}.mp4")
    _save_frames_(file_path, combined_images)
    return file_path

def group_video_of_clusters(cluster_assignments, frames_with_labels, run_desc, epochs,
                            n_sequences_to_draw=9, n_clusters_to_draw=10):
    grouped = group_by_cluster(cluster_assignments)

    sorted_groups = sorted([(g, sorted(vals, key=len, reverse=True)) for g, vals in grouped.items()],
                           key=lambda x: max(map(len, x[1])),
                           reverse=True)

    for cluster_id, sequences in sorted_groups[:n_clusters_to_draw]:
        sequences[:n_sequences_to_draw]
        paths = [[_path_for_image_(image_id, label) for image_id, label in frames_with_labels[seq]] for seq in sequences]
        #return paths
        yield cluster_id, group_video_of_cluster(cluster_id, paths, run_desc, epochs=epochs, n_sequences_to_draw=n_sequences_to_draw)



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

ls ../../data/AlphaTheta/sam-AlphaTheta/offline/fif

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
