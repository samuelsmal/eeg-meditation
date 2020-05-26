import yaml
import os.path as path
from enum import Enum

from nfme.neurodecode_protocols.music import MusicMixStyle

class FeatureType(Enum):
    THETA = 1
    ALPHA_THETA = 2


class ProtocolConfig(dict):
    def __init__(self, file_path, defaults_file_path=None):
        dict.__init__(self)

        with open(file_path, 'r') as f:
            dct = yaml.safe_load(f)

        defaults_file_path = defaults_file_path or \
            path.abspath(path.join(path.dirname(__file__),
                                   f"../../protocol_configs/default/{dct['protocol_type']}.yml"))

        with open(defaults_file_path, 'r') as f:
            self.update(yaml.safe_load(f))

        self.update(dct)

        for k, v in self.items():
            if isinstance(v, str) and '_root_music_path_' in v:
                self[k] = v.format(**{'_root_music_path_': self['_root_music_path_']})

            if isinstance(v, str) and '_root_data_path_' in v:
                self[k] = v.format(**{'_root_data_path_': self['_root_data_path_']})

        self['music_mix_style'] = MusicMixStyle[self['music_mix_style']]
        self['feature_type'] = FeatureType[self['feature_type']]

        self['protocol_name'] = file_path.split('/')[-1].split('.')[0]

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

if __name__ == '__main__':
    _t = path.abspath(path.join(path.dirname(__file__), '../protocol_configs/dvorak_study.yml'))
    print(ProtocolConfig(_t))
