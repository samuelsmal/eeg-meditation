# 3p
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter

# project
from src import helpers
from src.configuration import cfg


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_filtered_signal(
    bands, data_type="meditation", subject="sam", recording=0, group_time="10s"
):
    df = helpers.load_signal_data(data_type, cfg, subject=subject, recording=recording)
    df["avg"] = df.mean(axis=1)

    dfs = [
        df.apply(
            lambda x: butter_bandpass_filter(
                x,
                cfg["bands"][band][0],
                cfg["bands"][band][1],
                cfg["sampling_frequency"],
            )
        ).mean(axis=1)
        for band in bands
    ]

    merged = (
        pd.concat(dfs, keys=bands)
        .unstack(0)
        .groupby([pd.Grouper(freq=group_time)])
        .sum()
        / 10
    )
    print(merged.head())

    plt.figure()
    merged.plot()

    plt.title(
        "Average signal, bandpass filtered in band {} for\n recording {} and subject {}, idx {}".format(
            bands, data_type, subject, recording
        )
    )
    plt.legend()
    plt.show()


def plot_concurrent_signals(
    meditation, baseline, title="", band="alpha", group_time="4s", electrode="avg"
):
    meditation["avg"] = meditation.mean(axis=1)
    baseline["avg"] = baseline.mean(axis=1)

    filtered_meditation = (
        meditation.apply(
            lambda x: butter_bandpass_filter(
                x,
                cfg["bands"][band][0],
                cfg["bands"][band][1],
                cfg["sampling_frequency"],
            )
        )
        .groupby([pd.Grouper(freq=group_time)])
        .mean()
    )

    filtered_baseline = (
        baseline.apply(
            lambda x: butter_bandpass_filter(
                x,
                cfg["bands"][band][0],
                cfg["bands"][band][1],
                cfg["sampling_frequency"],
            )
        )
        .groupby([pd.Grouper(freq=group_time)])
        .mean()
    )

    merged = pd.merge(
        filtered_baseline.reset_index()[electrode],
        filtered_meditation.reset_index()[electrode],
        how="outer",
        left_index=True,
        right_index=True,
    ).rename(
        columns={
            "{}_x".format(electrode): "baseline",
            "{}_y".format(electrode): "meditation",
        }
    )

    plt.figure()
    merged.plot()
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_type = "baseline"
    subject = "adelie"
    band = "theta"
    recording = 1

    # meditation = helpers.load_signal_data(
    #     "meditation", cfg, subject=subject, recording=recording
    # )
    # baseline = helpers.load_signal_data(
    #     "baseline", cfg, subject=subject, recording=recording
    # )

    # plot_concurrent_signals(meditation, baseline, band)
    plot_filtered_signal(
        ["alpha", "theta", "gamma"],
        data_type=data_type,
        subject=subject,
        recording=recording,
        group_time="10s",
    )
