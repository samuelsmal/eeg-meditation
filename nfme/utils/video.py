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

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mne import viz
import numpy as np
import cv2
import warnings
import imageio

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from nfme.utils import loading_bar

def create_video(data, times_index_to_plot, pos, file_name):
    # create OpenCV video writer
    video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('A','V','C','1'), 1, (250, 250))
    bandpower_over_time_index = data.index

    # loop over your images
    for idx, time_index in enumerate(times_index_to_plot):
        fig = plt.figure()
        viz.plot_topomap(data.iloc[time_index, :].T.values,
                         sphere=1.,
                         pos=pos,
                         sensors=False,
                         show_names=True,
                         show=False,
                         names=data.columns)

        plt.title(index_to_time(times_index_to_plot[idx], bandpower_over_time_index))
        fig.canvas.draw()

        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        # write frame to video
        video.write(mat)
        plt.close()

    # close video writer
    cv2.destroyAllWindows()
    video.release()

def index_to_time(x, time_index, step_size=1):
    """Helper function to add the axis labels"""
    if (x < 0 or x * step_size >= len(time_index)):
        return ''

    seconds = time_index[int(x*step_size)].total_seconds()
    return f"{int(seconds/60)}\' {seconds/60:.2f}\""

def gen_topomap_frames(data, times_index_to_plot, pos, title, cmap='PuBu'):
    # create OpenCV video writer

    bandpower_over_time_index = data.index

    frames = []


    cNorm = mpl.colors.Normalize(vmin=data.min().min(), vmax=data.max().max())
    sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    # loop over your images
    for idx, time_index in enumerate(times_index_to_plot):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        viz.plot_topomap(data.iloc[time_index, :].T.values,
                         cmap=cmap,
                         sphere=1.,
                         pos=pos,
                         axes=axs,
                         sensors=False,
                         show_names=True,
                         show=False,
                         names=data.columns)

        plt.title(f"{title}: {index_to_time(times_index_to_plot[idx], bandpower_over_time_index)}")

        #cax = fig.add_axes([1.01, 0.2, 0.05, 0.5])
        # notice that this will create some weird plot if you plot it interactively... no idea why
        fig.colorbar(sm, ax=axs, orientation='vertical')

        plt.tight_layout()

        # this has to be the last thing
        fig.canvas.draw()

        mat = np.array(fig.canvas.renderer._renderer)
        #mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        frames.append(mat)

        plt.close()

    return frames

def save_frames(frames, file_name, fps=10):
    """uses imageio to save the given frames

    does only store `mp4` videos for now

    frames: list of frames, each frame has to be a 3d array: [height, width, channels]
    """

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16')
        imageio.mimsave(file_name, frames, format='mp4', fps=fps)

def normalise_layout_pos(layout):
    _t = layout.pos[:, :2]
    _t -= np.mean(_t, 0)
    _t /= np.max(_t, 0)

    return _t


def gen_topomap_video(data, normalised_pos, title, fraction_to_plot= 0.01):
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


def gen_topomap_frames_all_bands(data_all_bands, pos, cmap='PuBu', fraction_to_plot=0.01):
    # create OpenCV video writer

    bandpower_over_time_index = data_all_bands[list(data_all_bands.keys())[0]].index
    n_plots = np.int(bandpower_over_time_index.shape[0] * fraction_to_plot) # // cfg['sampling_frequency']
    times_index_to_plot = np.linspace(start=0, stop=bandpower_over_time_index.shape[0] - 1, num=n_plots, dtype=np.int)

    frames = []

    scalar_mappable_for_band = {b: cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=data.min().min(),
                                                                               vmax=data.max().max()),
                                                     cmap=cmap)
                                for b, data in data_all_bands.items()}
    n_cols = len(scalar_mappable_for_band.keys())

    ratios_to_compute = [('gamma', 'beta'),
                         ('gamma', 'alpha'),
                         ('beta', 'alpha'),
                         ('beta', 'theta'),
                         ('alpha', 'theta')]

    n_rows = 2
    # loop over your images

    ratios = {r: data_all_bands[r[0]] / data_all_bands[r[1]] for r in ratios_to_compute}

    print('generating frames...')
    n_frames_to_generate = len(times_index_to_plot)

    for idx, time_index in enumerate(times_index_to_plot):
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10 * n_cols, 10))
        for ax_idx, (band, data) in enumerate(data_all_bands.items()):
            viz.plot_topomap(data.iloc[time_index, :].T.values,
                             cmap=cmap,
                             sphere=1.,
                             pos=pos,
                             axes=axs[0][ax_idx],
                             sensors=False,
                             show_names=True,
                             show=False,
                             names=data.columns)

            axs[0][ax_idx].set_title(f"{band}: {index_to_time(times_index_to_plot[idx], bandpower_over_time_index)}")

            # notice that this will create some weird plot if you plot it interactively... no idea why
            fig.colorbar(scalar_mappable_for_band[band],
                         ax=axs[0][ax_idx],
                         orientation='vertical')

        for ax_idx, (ratio, data) in enumerate(ratios.items()):
            viz.plot_topomap(data.iloc[time_index, :].T.values,
                             cmap=cmap,
                             sphere=1.,
                             pos=pos,
                             axes=axs[1][ax_idx],
                             sensors=False,
                             show_names=True,
                             show=False,
                             names=data.columns)

            axs[1][ax_idx].set_title(f"{ratio}: {index_to_time(times_index_to_plot[idx], bandpower_over_time_index)}")

        plt.tight_layout()

        # this has to be the last thing
        fig.canvas.draw()

        mat = np.array(fig.canvas.renderer._renderer)
        #mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        frames.append(mat)

        plt.close()

        loading_bar(idx, n_frames_to_generate)

    print('all frames generated')

    return frames
