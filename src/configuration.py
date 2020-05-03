import numpy as np
from functools import reduce

cfg = {
    "paths": {
        "base": "C:/Users/adeli/OneDrive/Documents/GitHub/data/AlphaTheta",
        "subjects": {
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


def get_config_value(config, *args):
    """Helper to get read the config"""
    return reduce(lambda config, val: config[val], args, config)
