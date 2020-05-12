#----------------------------------------------------------------------
# Parameters to define
#----------------------------------------------------------------------
DATA_PATH = ''

SPATIAL_FILTER = None
SPATIAL_CHANNELS = ['Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'P3', 'Pz', 'P4']

ALPHA_CHANNELS = ['Pz']
ALPHA_REF = 239550
ALPHA_THR = 376355

ALPHA_FREQ = {'min': 8, 'max': 12}
THETA_FREQ = {'min': 4, 'max': 8}

THETA_CHANNELS = ['Pz']
THETA_REF = 702870
THETA_THR = 982914

STREAMBUFFER = 2                   # Stream buffer [sec]
WINDOWSIZE = 2                     # window length of acquired data when calling get_window [sec]

TIMER_SLEEP = 0.01 * 60

GLOBAL_TIME = 300

NJOBS = 8                          # For multicore PSD processing

_ROOT_MUSIC_PATH_ = '/home/sam/proj/epfl/eeg-meditation-project/nfme/meditation_sounds/'

HARMONIC_SOUND_PATH =   f"{_ROOT_MUSIC_PATH_}/harmonisch.wav"
UNHARMONIC_SOUND_PATH = f"{_ROOT_MUSIC_PATH_}/unharmonisch.wav"

SCRIPT_BEAT_HARMONY_MELODY_PATH = f"{_ROOT_MUSIC_PATH_}/dvorak_study/script_beat_harmony_melody.wav"
SCRIPT_BEAT_HARMONY_PATH = f"{_ROOT_MUSIC_PATH_}/dvorak_study/script_beat_harmony.wav"

SCRIPT_BEAT_HARMONY_MELODY_WITHOUT_SCRIPT_PATH = f"{_ROOT_MUSIC_PATH_}/dvorak_study/script_beat_harmony_melody_without_script.wav"
SCRIPT_BEAT_HARMONY_WITHOUT_SCRIPT_PATH = f"{_ROOT_MUSIC_PATH_}/dvorak_study/script_beat_harmony_without_script.wav"

MUSIC_STATE_1_PATH = SCRIPT_BEAT_HARMONY_MELODY_WITHOUT_SCRIPT_PATH
MUSIC_STATE_2_PATH = SCRIPT_BEAT_HARMONY_WITHOUT_SCRIPT_PATH

MUSIC_WINDOW_SIZE = 10
