# 3p
import pandas as pd
import matplotlib.pyplot as plt

# project
from src.configuration import cfg
from src import helpers


def plot_bandpower_bar(bp, bands_to_plot=cfg["bands"].keys(), title="Bars"):
    mean_bp = bp.mean(level=1, axis=0)
    print(mean_bp)
    bands_to_plot_filter = mean_bp.index.get_level_values(0).isin(bands_to_plot)
    mean_bp[bands_to_plot_filter].plot.bar(rot=0)

    plt.ylabel("Mean spectral power (µV²/Hz)")

    plt.title(title)
    plt.legend()
    plt.show()


def plot_bandpower_bar_std_concurrent(
    merged, bands_to_plot=cfg["bands"].keys(), title="Bars and std"
):
    mean_bp = merged.mean(axis=1)
    aggregated_bp = mean_bp.groupby(level=[0, 1]).agg(["mean", "std"])

    bands_to_plot_filter = aggregated_bp.index.get_level_values(1).isin(bands_to_plot)

    aggregated_bp[bands_to_plot_filter]["mean"].unstack(0).plot.bar(
        rot=0, yerr=aggregated_bp["std"].unstack(0)
    )

    plt.ylabel("Mean spectral power (µV²/Hz)")

    plt.title(title)
    plt.legend()
    plt.show()


def plot_bandpower_line(bp, title=""):
    """
    WARNING: x-axis is time, which is not robust
    Parameters
    ----------
    bp: pd.DataFrame
    title: str

    Returns
    -------

    """
    bp.plot()

    plt.ylabel("Mean spectral power (µV²/Hz)")
    plt.xlabel("Epochs")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_baseline_vs_meditation_average_band(
    bands="alpha", subject="adelie", recording=1, **kwargs
):
    baseline = helpers.load_bandpower_all_epochs_df(
        "baseline", config=cfg, subject=subject, recording=recording, **kwargs
    )
    meditation = helpers.load_bandpower_all_epochs_df(
        "meditation", config=cfg, subject=subject, recording=recording, **kwargs
    )

    baseline["avg"] = baseline.mean(axis=1)
    meditation["avg"] = meditation.mean(axis=1)

    merged = pd.merge(
        baseline.loc[bands].reset_index()["avg"],
        meditation.loc[bands].reset_index()["avg"],
        how="outer",
        left_index=True,
        right_index=True,
    ).rename(columns={"avg_x": "baseline", "avg_y": "meditation"})

    plot_bandpower_line(
        merged,
        title="Average bandpower at band {} as a function of epochs".format(bands),
    )


def plot_baseline_vs_meditation_average_bands_electrode(
    electrode, bands="alpha", subject="adelie", recording=1, **kwargs
):
    baseline = helpers.load_bandpower_all_epochs_df(
        "baseline", config=cfg, subject=subject, recording=recording, **kwargs
    )
    meditation = helpers.load_bandpower_all_epochs_df(
        "meditation", config=cfg, subject=subject, recording=recording, **kwargs
    )

    merged = pd.merge(
        baseline.loc[bands].reset_index()[electrode],
        meditation.loc[bands].reset_index()[electrode],
        how="outer",
        left_index=True,
        right_index=True,
    ).rename(columns={f"{electrode}_x": "baseline", f"{electrode}_y": "meditation"})

    plot_bandpower_line(
        merged,
        title="Bandpower on electrode {} at band {} as a function of epochs".format(
            electrode, bands
        ),
    )
