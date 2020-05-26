# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import yaml

# <codecell>

with open('../neurodecode_protocols/meditation/sam-meditation/online.yml', 'r') as f:
    cfg = yaml.safe_load(f)

# <codecell>



# <codecell>

cfg

# <codecell>



# <codecell>

m.group()

# <codecell>

import warnings
import re

class ProtocolConfig(dict):
    def __init__(self, file_name):
        dict.__init__(self)
        
        with open(file_name, 'r') as f:
            cfg = yaml.safe_load(f)
            
        for k, v in cfg.items():
            if isinstance(v, str) and (m := re.match('\{(.*?)\}', v)):
                to_replace = m.group()[1:-1]

                if (to_replace_val := cfg.get(to_replace, None)):
                    print(f"replacing '{to_replace}' with '{to_replace_val}'")
                    cfg[k] = v.format(**{to_replace: to_replace_val})
                else:
                    warnings.info(f"could not replace {to_replace} as the key doesn't exist in the config")
            
        self.update(cfg)

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

# <codecell>

cfg['script_beat_harmony_melody_path'].format(_ROOT_MUSIC_PATH_=21)

# <codecell>

pcfg = ProtocolConfig('../neurodecode_protocols/meditation/sam-meditation/online.yml')

# <codecell>

pcfg

# <codecell>

if (a := 0) :
    print('ere')
    
print(a)

# <codecell>


