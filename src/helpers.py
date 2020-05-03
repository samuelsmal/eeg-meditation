# 3p
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal

# project
from src.configuration import get_config_value, cfg
from src import bandpower as bp


def load_signal_data(data_type, config, subject="sam", recording=0):
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
    subject_paths = get_config_value(config, "paths", "subjects", subject)
    data = pd.read_pickle(
        f"{config['paths']['base']}/{subject_paths['prefix']}/offline/{get_config_value(subject_paths, 'recordings', data_type)[recording]}-raw.pcl"
    )

    _t = data["timestamps"].reshape(-1)
    _t -= _t[0]

    signal = pd.DataFrame(
        data=data["signals"],
        index=pd.TimedeltaIndex(_t, unit="s"),
        columns=data["ch_names"],
    ).drop(columns=config["columns_to_remove"])

    return signal.loc[signal.index[config["default_signal_crop"]], :]


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

    fig, axs = plt.subplots(
        nrows=signal_pd.shape[1], figsize=(40, 1.4 * signal_pd.shape[1]), sharex=True
    )
    for channel_id, channel in enumerate(signal_pd.columns):
        d = signal_pd.loc[::sampling, channel]
        sns.lineplot(data=d.reset_index(drop=True), ax=axs[channel_id])
        axs[channel_id].set_ylabel(channel)

    axs[-1].set_xlabel("time [ms]")
    axs[-1].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: index_to_time(x, d.index))
    )

    fig.suptitle("Raw signal recording")

    return fig


def index_to_time(x, time_index, step_size=1):
    """Helper function to add the axis labels"""
    if x < 0 or x * step_size >= len(time_index):
        return ""

    seconds = time_index[int(x * step_size)].total_seconds()
    return f"{int(seconds/60)}' {seconds/60:.2f}\""


def get_channels_list(config=cfg, subject="adelie"):
    subject_paths = get_config_value(config, "paths", "subjects", subject)
    base_path = get_config_value(config, "paths", "base")
    file_path = f"{base_path}/{subject_paths['prefix']}/offline/fif/{subject_paths['channels_path']}"
    with open(file_path, "r") as channels_file:
        all_channels = channels_file.read().strip()
    return [
        channel
        for channel in all_channels.split("\n")
        if channel not in config["columns_to_remove"]
    ]


def load_bandpower_epoch_df(
    data_type,
    start,
    end,
    subject="adelie",
    recording=0,
    remove_references=True,
    config=cfg,
    window_size="2s",
):
    df = load_signal_data(
        data_type, subject=subject, recording=recording, config=config
    )[start:end]
    electrodes = get_channels_list(config, subject)
    return bp.get_all_electrodes_bandpowers_df(df, electrodes, window_size=window_size)


def load_bandpower_all_epochs_df(
    data_type, subject="adelie", recording=0, config=cfg, **kwargs
):
    df = load_signal_data(
        data_type, subject=subject, recording=recording, config=config
    )
    return bp.get_bandpower_epochs_for_all_electrodes(df, config=cfg, **kwargs)


def preprocessing(series, center=True, reduce=True, detrend=True):
    if center:
        series = series - series.mean()
    if reduce:
        series = series / series.std()
    if detrend:
        series = signal.detrend(series)
    return series


def load_bandpower_all_epochs_all_recordings_df(
    data_type, subject="adelie", config=cfg, **kwargs
):
    dfs = [
        load_signal_data(data_type, subject=subject, recording=recording, config=config)
        for recording in range(
            len(config["paths"]["subjects"][subject]["recordings"][data_type])
        )
    ]

    for i in range(len(dfs) - 1):
        dfs[i + 1].index += dfs[i].index.max()

    concatenated = pd.concat(dfs)
    return bp.get_bandpower_epochs_for_all_electrodes(
        concatenated, config=cfg, **kwargs
    )
