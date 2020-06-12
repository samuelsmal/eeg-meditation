import numpy as np
import pandas as pd
from src.configuration import cfg
import src.helpers as helpers
import mne
from mne import viz
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib

# Get the ratio of bandpowers (baseline/meditation) for specific frequency band
def get_ratio_band(df1_bp, df2_bp, band="band", csv_name=""):
    # df1 and df2 are bandpowers
    df1_band = df1_bp.xs(band, level=0).mean(axis=0)
    df2_band = df2_bp.xs(band, level=0).mean(axis=0)

    ratio_band = df2_band / df1_band
    ratio_band_log = pd.DataFrame(ratio_band.apply(np.log)).transpose()
    ratio_band_log.to_csv(csv_name)

    return ratio_band_log


# Get the average ratio of bandpowers (baseline/meditation) for specific frequency band
def get_avg_ratio(df1_bp, df2_bp, subj=""):
    # df1 and df2 are bandpowers
    mean_df1_bp = df1_bp.mean(axis=1).groupby(level=0).mean()
    mean_df2_bp = df2_bp.mean(axis=1).groupby(level=0).mean()

    ratio_bp = mean_df2_bp / mean_df1_bp
    log_ratio_bp = pd.DataFrame(ratio_bp.apply(np.log))

    return log_ratio_bp.rename(columns={0: subj})


# Get mean and svd frequency (baseline and meditation) for one specific frequency band
def get_mean_svd_band(df1_psd, df2_psd, band="band"):
    # df1 and df2 are welch signals
    df1_psd_filtered = (
        df1_psd[
            (df1_psd.index.get_level_values(1) >= band[0])
            & (df1_psd.index.get_level_values(1) <= band[1])
        ]
        .groupby(level=0)
        .idxmax()
        .applymap(lambda x: x[1])
    )
    df2_psd_filtered = (
        df2_psd[
            (df2_psd.index.get_level_values(1) >= band[0])
            & (df2_psd.index.get_level_values(1) <= band[1])
        ]
        .groupby(level=0)
        .idxmax()
        .applymap(lambda x: x[1])
    )

    df1 = pd.DataFrame(df1_psd_filtered.agg(["mean", "std"])["AVG"]).rename(
        columns={"AVG": "baseline"}
    )
    df2 = pd.DataFrame(df2_psd_filtered.agg(["mean", "std"])["AVG"]).rename(
        columns={"AVG": "meditation"}
    )
    df = pd.concat([df1, df2], axis=1)

    return df


# Boxplot
def get_boxplot_band(df1_psd, df2_psd, band="band", title=""):

    # df1 and df2 are welch signals
    df1_psd_filtered = (
        df1_psd[
            (df1_psd.index.get_level_values(1) >= band[0])
            & (df1_psd.index.get_level_values(1) <= band[1])
        ]
        .groupby(level=0)
        .idxmax()
        .applymap(lambda x: x[1])
    )
    df2_psd_filtered = (
        df2_psd[
            (df2_psd.index.get_level_values(1) >= band[0])
            & (df2_psd.index.get_level_values(1) <= band[1])
        ]
        .groupby(level=0)
        .idxmax()
        .applymap(lambda x: x[1])
    )
    df_concat = pd.concat(
        [df1_psd_filtered["AVG"], df2_psd_filtered["AVG"]],
        keys=["baseline", "meditation"],
    )
    plot = (
        df_concat.reset_index(level=0)
        .rename(columns={"level_0": "Recording type"})
        .boxplot(
            by="Recording type",
            boxprops=dict(linestyle="-", linewidth=4, color="k"),
            medianprops=dict(linestyle="-", linewidth=4, color="k"),
        )
    )
    plot.get_figure().suptitle(title)

    return plot


def get_concat_bandpower(data_type, subject="adelie", config=cfg):
    dfs = [
        helpers.load_bandpower_all_epochs_df(
            data_type, subject=subject, recording=recording, config=config
        ).swaplevel(0, 1)
        for recording in range(
            len(config["paths"]["subjects"][subject]["recordings"][data_type])
        )
    ]

    for i in range(len(dfs) - 1):
        new_time_level = (
            dfs[i + 1].index.levels[0] + dfs[i].index.get_level_values(0).max()
        )
        dfs[i + 1].index.set_levels(new_time_level, level=0, inplace=True)

    concatenated = pd.concat(dfs)
    return concatenated


def select_band_for_electrodes(electrodes, band="theta", subject="adelie"):
    baseline = get_concat_bandpower("baseline", config=cfg, subject=subject)
    meditation = get_concat_bandpower("meditation", config=cfg, subject=subject)

    baseline_band_electrodes = baseline.xs(band, level=1)[electrodes]
    meditation_band_electrodes = meditation.xs(band, level=1)[electrodes]

    merged = pd.concat(
        [baseline_band_electrodes, meditation_band_electrodes],
        keys=["baseline", "meditation"],
    )

    return merged


def get_concat_aligned_bandpower(pre_merge_meditation, pre_merge_baseline):
    pre_merge_meditation.index = pre_merge_meditation.index.set_levels(
        pd.RangeIndex(len(pre_merge_meditation.index)), level=1
    )
    pre_merge_baseline.index = pre_merge_baseline.index.set_levels(
        pd.RangeIndex(len(pre_merge_baseline.index)), level=1
    )

    concat_aligned = pd.concat(
        [pre_merge_baseline, pre_merge_meditation], keys=["baseline", "meditation"]
    )

    return concat_aligned
