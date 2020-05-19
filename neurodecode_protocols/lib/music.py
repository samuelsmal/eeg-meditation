from enum import Enum

class MusicMixStyle(Enum):
    REPLACING = 1
    ADDITIVE_NEGATIVE = 2
    ADDITIVE_POSITIVE = 3

def mix_sounds(style:MusicMixStyle, sounds, feature_value):
    s1, s2 = sounds

    if style == MusicMixStyle.REPLACING:
        s1.set_volume(feature_value)
        s2.set_volume(1 - feature_value)
    elif style == MusicMixStyle.ADDITIVE_NEGATIVE:
        s2.set_volume(1 - feature_value)
    elif style == MusicMixStyle.ADDITIVE_POSITIVE:
        s2.set_volume(feature_value)
    else:
        raise ValueError('MusicMixStyle not supported')
