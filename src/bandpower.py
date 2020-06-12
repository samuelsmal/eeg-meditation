# std
import time
from itertools import combinations

# 3p
import matplotlib as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper

# project
from src import helpers
from src.configuration import cfg


def bandpower(data, sf, band, method="welch", window_sec=None, relative=False):
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
    if method == "welch":
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        if data.shape[0] < nperseg:
            return np.NaN

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == "multitaper":
        psd, freqs = psd_array_multitaper(
            data, sf, adaptive=True, normalization="full", verbose=0
        )

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)

    return bp


def get_bandpower_for_electrode(signal_data, electrode, config, window_sec=2):
    """Calculates the bandpower for the given electrode

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
    for band_name, band_range in config["bands"].items():
        xs = signal_data.loc[:, electrode]
        bandpowers[band_name] = bandpower(xs, config["sampling_frequency"], band_range)

    # compute all different ratios
    for bn_l, bn_r in combinations(config["bands"].keys(), 2):
        bandpowers[f"{bn_l} / {bn_r}"] = bandpowers[bn_l] / bandpowers[bn_r]
    return bandpowers


def get_bandpower_epochs_for_all_electrodes(
    signal_data, config, epoch_size="10s", window_sec=2
):
    """Calculates the bandpower for the given electrode for different epochs

    Parameters
    ----------
    signal_data: pd.DataFrame
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
    for band_name, band_range in config["bands"].items():
        bandpowers[band_name] = signal_data.groupby(
            [pd.Grouper(freq=epoch_size)]
        ).aggregate(
            lambda epoch: bandpower(
                helpers.preprocessing(epoch),
                # epoch,
                config["sampling_frequency"],
                band_range,
                window_sec=window_sec,
            )
        )

    # compute all different ratios
    for bn_l, bn_r in combinations(config["bands"].keys(), 2):
        bandpowers[f"{bn_l} / {bn_r}"] = bandpowers[bn_l] / bandpowers[bn_r]

    result = pd.concat(list(bandpowers.values()), keys=list(bandpowers.keys()))
    for column in result.columns:
        result[column] = result[column].astype(float)
    return result


def get_bandpower_epochs_for_all_electrodes_v2(
    signal_data,
    sampling_frequency,
    bands,
    epoch_size="10s",
    window_sec=2,
    apply_preprocessing=False,
    target_level=None,
):
    """Calculates the bandpower for the given electrode for different epochs

    Parameters
    ----------
    signal_data: pd.DataFrame
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
    groups = [pd.Grouper(level=0), pd.Grouper(freq=epoch_size, level=target_level)]
    for band_name, band_range in bands.items():
        bandpowers[band_name] = signal_data.groupby(groups).aggregate(
            lambda epoch: bandpower(
                helpers.preprocessing(epoch) if apply_preprocessing else epoch,
                sampling_frequency,
                band_range,
                window_sec=window_sec,
            )
        )

    # compute all different ratios
    for bn_l, bn_r in combinations(bands.keys(), 2):
        bandpowers[f"{bn_l} / {bn_r}"] = bandpowers[bn_l] / bandpowers[bn_r]

    result = pd.concat(list(bandpowers.values()), keys=list(bandpowers.keys()))
    for column in result.columns:
        result[column] = result[column].astype(float)
    return result


def get_all_electrodes_bandpowers(df, electrodes, config=cfg, window_sec=2):
    start = time.time()
    all_bandpowers = {}
    for electrode in electrodes:
        all_bandpowers[electrode] = get_bandpower_for_electrode(
            df, electrode=electrode, config=config, window_sec=window_sec
        )

    end = time.time()
    print("Took {}s to compute bandpower for all electrodes".format(end - start))
    return all_bandpowers


def get_all_electrodes_bandpowers_df(df, electrodes, config=cfg, window_sec=2):
    all_bandpowers_dict = get_all_electrodes_bandpowers(
        df, electrodes, config, window_sec=window_sec
    )
    result_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                electrodes,
                all_bandpowers_dict[list(all_bandpowers_dict.keys())[0]].keys(),
            ],
            names=["electrodes", "bands/ratios"],
        ),
        columns=["values"],
    )
    for electrode, bandpowers in all_bandpowers_dict.items():
        for band, power in bandpowers.items():
            result_df.loc[(electrode, band), :] = power
    result_df["values"] = result_df["values"].astype(float)
    return result_df


def aggregate_bandpower(baseline, signal):
    aggregated_fns = ["mean", "median", "min", "max"]
    aggregated_power = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [list(baseline.keys()), ["baseline", "meditation"]]
        ),
        columns=aggregated_fns,
    )

    for band, power in baseline.items():
        aggregated_power.loc[(band, "baseline"), :] = power.agg(aggregated_fns)

    for band, power in signal.items():
        aggregated_power.loc[(band, "meditation"), :] = power.agg(aggregated_fns)

    return aggregated_power


def plot_bandpowers(bandpowers, electrode):
    fig, axs = plt.subplots(nrows=len(bandpowers), sharex=True, figsize=(25, 15))
    time_index = list(bandpowers.values())[0].index
    time_index_as_seconds = [t.total_seconds() for t in time_index]

    for i, (bn, bp) in enumerate(bandpowers.items()):
        axs[i].plot(bp.reset_index(drop=True))
        axs[i].set_ylabel(bn)

    axs[0].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: helpers.index_to_time(x, time_index))
    )
    fig.suptitle(f"Bandpower of {electrode}")

    return fig
