import numpy as np
from functools import reduce

cfg = {
    "paths": {
        "base": "C:/Users/adeli/OneDrive/Documents/GitHub/data/AlphaTheta",
        "subjects": {
            ### RECORDINGS MARCH
            "sam": {
                "prefix": "/sam-AlphaTheta",
                "recordings": {
                    "baseline": ["baseline11", "baseline2", "baseline10"],
                    "meditation": ["meditation1", "meditation2"],
                },
                "channels_path": "channelsList.txt",
            },
            "adelie": {
                "prefix": "/adelie-AlphaTheta",
                "recordings": {
                    "baseline": ["baseline1", "baseline2"],
                    "meditation": ["meditation1", "meditation2"],
                },
                "channels_path": "channelsList.txt",
            },
            ### RECORDINGS MAY 1
            "sam2": {
                "prefix": "/recording_2020-05-27",
                "recordings": {
                    "baseline": ["sam-baseline"],
                    "meditation": ["sam-meditation"],
                },
                "channels_path": "channelsList.txt",
            },
            "raphael": {
                "prefix": "/recording_2020-05-27",
                "recordings": {
                    "baseline": ["rap-baseline"],
                    "meditation": ["rap-meditation"],
                },
                "channels_path": "channelsList.txt",
            },
            ### RECORDINGS MAY 2
            "sam3": {
                "prefix": "/recording_2020-05-29",
                "recordings": {
                    "baseline": ["sam-baseline"],
                    "meditation": ["sam-meditation"],
                },
                "channels_path": "channelsList.txt",
            },
            "raphael2": {
                "prefix": "/recording_2020-05-29",
                "recordings": {
                    "baseline": ["rap1-baseline"],
                    "meditation": ["rap1-meditation"],
                },
                "channels_path": "channelsList.txt",
            },
            "raphael3": {
                "prefix": "/recording_2020-05-29",
                "recordings": {
                    "baseline": ["rap2-baseline"],
                    "meditation": ["rap2-meditation"],
                },
                "channels_path": "channelsList.txt",
            },
            "arn": {
                "prefix": "/recording_2020-05-29",
                "recordings": {
                    "baseline": ["arn-baseline"],
                    "meditation": ["arn-meditation"],
                },
                "channels_path": "channelsList.txt",
            },
            ### RECORDINGS JUNE
            "sam4": {
                "prefix": "/recording_2020-06-01",
                "recordings": {
                    "baseline": ["sam-baseline"],
                    "meditation": ["sam-meditation"],
                },
                "channels_path": "channelsList.txt",
            },
        },
    },
    "columns_to_remove": ["TRIGGER", "X1", "X2", "X3",],
    "default_signal_crop": np.s_[
        3000:-3000
    ],  # this corresponds to ~1 second at the beginning and end, given by the sampling frequency
    "sampling_frequency": 300,
    "bands": {
        "gamma": [40, 100],
        "beta": [12, 40],
        "alpha": [8, 12],
        "theta": [4, 8],
        "delta": [1, 4],
    },
}

cfg_mr = {
    "paths": {
        "base": "C:/Users/adeli/OneDrive/Documents/GitHub/data/AlphaTheta",
        "subjects": {
            "mr": {
                "prefix": "/MatthieuRicard",
                "recordings": {
                    "baseline": ["baseline1", "baseline2"],
                    "meditation": ["meditation"],
                },
            },
            "mr2": {
                "prefix": "/MatthieuRicard",
                "recordings": {
                    "baseline": ["baseline1.2", "baseline2.2"],
                    "meditation": ["meditation.2"],
                },
            },
        },
    },
    "columns_to_remove": ["nois", "sync", "STI 014"],
    "default_signal_crop": np.s_[
        3000:-3000
    ],  # this corresponds to ~1 second at the beginning and end, given by the sampling frequency
    "sampling_frequency": 250,
    "bands": {
        "gamma": [40, 100],
        "beta": [12, 40],
        "alpha": [8, 12],
        "theta": [4, 8],
        "delta": [1, 4],
    },
}


def get_config_value(config, *args):
    """Helper to get read the config"""
    return reduce(lambda config, val: config[val], args, config)
