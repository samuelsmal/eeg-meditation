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

"""
This defines a offline config. Meaning that you can't have two different configs.
The object `config` will be imported everywhere and used.

Parameters which are "adaptable", such as `reference_electrode` can always be provided as a
parameter to the functions. Only the paths are fixed (as they are also not expected to change).
"""

from functools import reduce
from pathlib import Path

import numpy as np

# This is the config. Add any value that you deem necessary. This should contain everything that belongs to the setup, the filtering pipeline, etc.

_ROOT_PATH_TO_MODULE_ = str(Path(__file__).parent.parent)

DEFAULT_VALUES = {
    'paths': {
        'base': _ROOT_PATH_TO_MODULE_ + '/data/AlphaTheta',
        'channels_file': 'channelsList.txt',
        'subjects': {
            'sam': {
                'prefix': '/sam-AlphaTheta',
                'recordings': {
                    'baseline': [
                        '20200304-144100',
                        '20200311-111333',
                    ],
                    'meditation': [
                        '20200304-144933',
                        '20200311-111621',
                    ]
                }
            },
            'adelie': {
                'prefix': '/adelie-AlphaTheta',
                'recordings': {
                    'baseline': [
                        '20200304-151358',
                        '20200311-104132',
                    ],
                    'meditation': [
                        '20200304-152058',
                        '20200311-105005',
                    ]
                }
            }
        },
        'features': _ROOT_PATH_TO_MODULE_ + '/data/features',
    },
    'columns_to_remove': [
        'TRIGGER', 'X1', 'X2', 'X3',
    ],
    'reference_electrode': 'A2',
    'default_signal_crop': np.s_[3000:-3000], # this corresponds to ~1 second at the beginning and end, given by the sampling frequency
    'bandpower_window_width': '1s',
    'sampling_frequency': 300,
    'bands': {
        'gamma': [40, 100],
        'beta':  [12, 40],
        'alpha': [8, 12],
        'theta': [4, 8],
        'delta': [0, 4]
    },
    'bandpass_filter': [0.5, 30]
}


class _Config_(dict):
    def __init__(self, **kwargs):
        dict.__init__(self)
        self.update(kwargs)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def value(self, *keys):
        """Helper to return one single value (may be what ever that value is).

        E.g. value('hubert', 'fly_id')
        """
        try:
            val = reduce(lambda d, k: d[k], keys, self)

            if isinstance(val, str) and 'UNKOWN' in val:
                raise ValueError('{0} value is not set! Reading {1}'.format(keys, val))
        except KeyError:
            raise ValueError("Could not find a value for the given key: {}".format(keys))

        return val


config = _Config_(**DEFAULT_VALUES)
