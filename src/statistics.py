import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.configuration import cfg


def get_information(df, epoch_size="10S"):
    meditation_correlations = df.groupby(pd.Grouper(freq=epoch_size)).corr()
    drop_nas = meditation_correlations.groupby(level=0).filter(
        lambda g: g.isnull().sum().sum() == 0
    )
    information = drop_nas.groupby(level=0).apply(
        lambda x: sum(np.linalg.eig(x)[0][:5])
    )
    return information


def get_norm_information(df, epoch_size="10S"):
    meditation_correlations = df.groupby(pd.Grouper(freq=epoch_size)).corr()
    drop_nas = meditation_correlations.groupby(level=0).filter(
        lambda g: g.isnull().sum().sum() == 0
    )
    information = drop_nas.groupby(level=0).apply(
        lambda x: sum(np.linalg.eig(x)[0][:5])
    )
    normalisation = drop_nas.groupby(level=0).apply(lambda x: sum(np.linalg.eig(x)[0]))
    return information / normalisation


def get_distrib_information(df, epoch_size="10S"):
    meditation_correlations = df.groupby(pd.Grouper(freq=epoch_size)).corr()
    drop_nas = meditation_correlations.groupby(level=0).filter(
        lambda g: g.isnull().sum().sum() == 0
    )
    distribution = drop_nas.groupby(level=0).apply(lambda x: np.linalg.eig(x)[0])
    normalisation = drop_nas.groupby(level=0).apply(lambda x: sum(np.linalg.eig(x)[0]))
    return distribution / normalisation


def get_dsp_db(raw_signal_df, only_positive=True, integration_time=10):
    # DSP = norm(TF)² * Integration_time = power spectral density
    raw_signal_df["AVG"] = raw_signal_df.mean(
        axis=1
    )  # add the mean over all electrodes
    fft = raw_signal_df.apply(np.fft.fft)
    dsp = fft.applymap(lambda x: np.linalg.norm(x) ** 2 * integration_time)
    dsp_db = dsp.applymap(lambda x: 10 * np.log10(x))
    dsp_db["freqs"] = np.fft.fftfreq(dsp_db.shape[0], d=1 / 300)
    if only_positive:
        dsp_db = dsp_db[
            dsp_db["freqs"] >= 0
        ]  # only get the positive part of the spectrum
    return dsp_db


def get_welch_perio(data, sf, window_sec=4, low=1, high=50):
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    if data.shape[0] < nperseg:
        return np.NaN
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return pd.Series(psd[idx_band], index=freqs[idx_band])


def get_dsp_welch(
    raw_signal_df, epoch_size="10s", welch_window_sec=4, config=cfg, **kwargs
):
    raw_signal_df["AVG"] = raw_signal_df.mean(
        axis=1
    )  # add the mean over all electrodes
    sf = config["sampling_frequency"]

    dsp = raw_signal_df.groupby(pd.Grouper(freq=epoch_size)).apply(
        lambda x: x.apply(
            lambda y: get_welch_perio(y, sf, window_sec=welch_window_sec, **kwargs)
        )
    )

    return (
        dsp.drop(columns=0)
        .groupby(level=0)
        .filter(lambda g: g.isnull().sum().sum() == 0)
    )


def get_slopes(raw_signal_df, with_welch=True):
    if raw_signal_df.shape[0] <= 0:
        return raw_signal_df
    if with_welch:
        dsp_db_pos = get_dsp_welch(raw_signal_df)
    else:
        dsp_db_pos = get_dsp_db(raw_signal_df, only_positive=True)
    slopes_df = dsp_db_pos.groupby(level=0).apply(
        lambda x: x.agg(
            lambda y: np.polyfit(x.index.get_level_values(1).astype(float), 1 / y, 1)[1]
        )
    )
    return slopes_df


def plot_dsp(
    raw_signal_df, title="Welch periodogram for all electrodes", electrode=None
):
    dsp = get_dsp_welch(raw_signal_df)
    if electrode is not None:
        dsp = dsp[electrode]
    plt.plot(
        dsp.index.get_level_values(1).astype(float), dsp,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spectral power density (µV²/Hz)")
    plt.legend()
    plt.title(title)
    plt.show()
