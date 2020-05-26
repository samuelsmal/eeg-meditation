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

from pathlib import Path
from functools import partial
from multiprocessing import Pool

import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper

from nfme.config import config

def bandpower_with_rolling_window(electrode, signal_data, window_size, sampling_frequency, band_range):
    # returning the electrode is important, the parallelism could in theory shuffle the electrodes
    # around
    return (electrode,
            signal_data.loc[:, electrode]\
                       .rolling(window_size)\
                       .apply(lambda xs: bandpower(xs, sampling_frequency, band_range)))

def compute_bandpowers(data, subject, recording_types=None, recording_ids=None, bands=None,
                       window_size='30s', sampling_frequency=None, force=False):
    """This is a helper method to compute the bandpower for all electrodes and all recording types
    ...

    Pay attention to required structure of `data`

    It should be a dict:
        {
            `recording_type`: [entry for each recording id],
        }

    Example:

        {
            'meditation': [raws_signals_session_1, raws_signals_session_2],
            'baseline': [raws_signals_session_1, raws_signals_session_2],
        }




    """

    fn_params = {
        'window_size': window_size,
        'sampling_frequency': sampling_frequency,
    }

    bandpowers = {}

    electrodes = list(data.values())[0][0].columns

    if recording_ids is None:
        recording_ids =  list(range(len(list(data.values()))))

    if recording_types is None:
        recording_types = list(data.keys())

    n_features_to_compute = len(electrodes) * len(recording_ids) * len(recording_types) * len(bands)

    print((f"In total {n_features_to_compute} features / bandpowers will have "
           "to be computed / loaded. This might take a while..."))

    n_features_computed = 0

    with Pool() as p:
        for recording_type in recording_types:
            bandpowers[recording_type] = {}
            for recording_id in recording_ids:
                bandpowers[recording_type][recording_id] = {}
                for band, band_range in bands.items():
                    file_path = (f"{config.value('paths', 'features')}/bandpowers"
                                 f"/{band}_over_time_{window_size}_{recording_type}"
                                 f"_{recording_id}_{subject}.pkl")

                    if Path(file_path).is_file() and not force:
                        bandpower_over_time = pd.read_pickle(file_path)
                    else:
                        fn = partial(bandpower_with_rolling_window,
                                     **{**fn_params,
                                        'band_range': band_range,
                                        'signal_data': data[recording_type][recording_id]})

                        bandpower_over_time =  pd.DataFrame(dict(p.map(fn, electrodes)))
                        bandpower_over_time.to_pickle(file_path)

                    bandpowers[recording_type][recording_id][band] = bandpower_over_time

                    n_features_computed += len(electrodes)

                    print(('computing bandpower: ' +
                           ('x' * int(n_features_computed / n_features_to_compute * 20)) +
                           ('.' * int((1 - n_features_computed / n_features_to_compute) * 20)) +
                           f" currently: {band}, {recording_id}, {recording_type}"),
                          end='\r')

                print(f"\nrecord nr {recording_id}, {recording_type} done")

    print(f"done with bandpower computation for subject {subject}")

    return bandpowers


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


def get_bandpower_for_electrode(signal_data, electrode, bands, sampling_frequency=300, window_size='1s'):
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


def aggregate_bandpower(baseline, signal):
    aggregated_fns = ['mean', 'median', 'min', 'max']
    aggregated_power = pd.DataFrame(index=pd.MultiIndex.from_product([list(baseline.keys()), ['baseline', 'meditation']]),
                                    columns=aggregated_fns)

    for band, power in baseline.items():
        aggregated_power.loc[(band, "baseline"), :] = power.agg(aggregated_fns)


    for band, power in signal.items():
        aggregated_power.loc[(band, 'meditation'), :] = power.agg(aggregated_fns)


    return aggregated_power
