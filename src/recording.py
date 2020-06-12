# std
from typing import List

# 3p
import pandas as pd
import numpy as np

# prj
import src.bandpower as bp
from src.configuration import cfg


class Recording:
    def __init__(
        self,
        data_type: str,
        path: str,
        sampling_frequency: int = 300,
        columns_to_remove: List[str] = None,
        default_crop_secs: float = 3,
    ):
        """

        Parameters
        ----------
        :data_type : str the type of recording `baseline` or `meditation`
        """
        self.data_type = data_type
        self.path = path
        self.sampling_frequency = sampling_frequency
        self.columns_to_remove = columns_to_remove
        self.default_crop_secs = default_crop_secs
        self._raw_signal = None

    def __add__(self, other):
        if self._raw_signal is None:
            self.load_signal()

        if self.sampling_frequency != other.sampling_frequency:
            Warning("The sampling frequencies are different, this is not good")

        new_path = (
            self.path.replace(".pcl", "")
            + "+"
            + other.path.replace(".pcl", "").split("/")[-1]
        )

        if self.data_type == other.data_type:
            res = Recording(
                self.data_type,
                new_path,
                sampling_frequency=self.sampling_frequency,
                columns_to_remove=list(
                    set(self.columns_to_remove + other.columns_to_remove)
                ),
                default_crop_secs=max(self.default_crop_secs, other.default_crop_secs),
            )
            other_signal = other.raw_signal
            other_signal.index += self._raw_signal.index.max()
            res.raw_signal = pd.concat([self.raw_signal, other_signal])
        else:
            print(
                Warning(
                    "You are trying to add 2 recordings that don't have the same datatype: {}!={},".format(
                        self.data_type, other.data_type
                    )
                )
            )
            res = Recording(
                "multi",
                new_path,
                sampling_frequency=self.sampling_frequency,
                columns_to_remove=list(
                    set(self.columns_to_remove + other.columns_to_remove)
                ),
                default_crop_secs=max(self.default_crop_secs, other.default_crop_secs),
            )
            res.raw_signal = pd.concat(
                [self.raw_signal, other.raw_signal],
                keys=[self.data_type, other.data_type],
            )
        return res

    @property
    def raw_signal(self):
        """loads the data and returns a pandas dataframe

        Parameters
        ----------

        Returns
        -------
        a pandas dataframe, timedeltaindexed of the raw signals
        """
        if self._raw_signal is None:
            self.load_signal()

        return self._raw_signal

    @raw_signal.setter
    def raw_signal(self, signal):
        self._raw_signal = signal

    def load_signal(self):
        data = pd.read_pickle(self.path)

        _t = data["timestamps"].reshape(-1)
        _t -= _t[0]

        signal = pd.DataFrame(
            data=data["signals"],
            index=pd.TimedeltaIndex(_t, unit="s"),
            columns=data["ch_names"],
        ).drop(columns=self.columns_to_remove)

        crop = np.s_[
            self.default_crop_secs
            * self.sampling_frequency : -self.default_crop_secs
            * self.sampling_frequency
        ]
        self._raw_signal = signal.loc[signal.index[crop], :]

    def bandpower_by_epoch(self, bands=cfg["bands"], epoch_size="10s", **kwargs):
        return bp.get_bandpower_epochs_for_all_electrodes_v2(
            self.raw_signal,
            self.sampling_frequency,
            bands,
            epoch_size=epoch_size,
            target_level=1 if self.data_type == "multi" else None,
            **kwargs
        )
