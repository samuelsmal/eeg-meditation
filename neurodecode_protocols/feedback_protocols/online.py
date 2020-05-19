#!/usr/bin/env python3
#coding:utf-8

import mne
import os
import sys
import pickle

import datetime
import cv2
import time
import json
import numpy as np
import multiprocessing as mp
import pygame.mixer as pgmixer

import neurodecode.utils.pycnbi_utils as pu

from neurodecode import logger
from neurodecode.utils import q_common as qc
from neurodecode.gui.streams import redirect_stdout_to_queue
from neurodecode.stream_receiver.stream_receiver import StreamReceiver
from neurodecode.protocols.viz_bars import BarVisual
from neurodecode.triggers.trigger_def import trigger_def
import neurodecode.triggers.pyLptControl as pyLptControl

from lib import utils as protocol_utils
from lib.config import ProtocolConfig
from lib.music import mix_sounds
from lib.keyboard_keys import KEYS

os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper
mne.set_log_level('ERROR')          # DEBUG, INFO, WARNING, ERROR, or CRITICAL

def add_to_queue(xs, x):
    xs[:-1] = xs[1:]
    xs[-1] = x
    return xs


def run(cfg, state=mp.Value('i', 1), queue=None, experiment_mode=True):
    """
    Online protocol for Alpha/Theta neurofeedback.
    """
    time_of_recording = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    redirect_stdout_to_queue(logger, queue, 'INFO')

    # Wait the recording to start (GUI)
    while state.value == 2: # 0: stop, 1:start, 2:wait
        pass

    # Protocol runs if state equals to 1
    if not state.value:
        sys.exit(-1)

    #----------------------------------------------------------------------
    # LSL stream connection
    #----------------------------------------------------------------------
    # chooose amp
    amp_name, amp_serial = protocol_utils.find_lsl_stream(state,
                                                          amp_name=cfg.get('amp_name', None),
                                                          amp_serial=cfg.get('amp_serial', None))

    # Connect to lsl stream
    sr = protocol_utils.connect_lsl_stream(amp_name=amp_name,
                                           amp_serial=amp_serial,
                                           window_size=cfg['window_size'],
                                           buffer_size=cfg['buffer_size'])

    # Get sampling rate
    sfreq = sr.get_sample_rate()

    # Get trigger channel
    trg_ch = sr.get_trigger_channel()

    #----------------------------------------------------------------------
    # PSD estimators initialization
    #----------------------------------------------------------------------
    psde_alpha = protocol_utils.init_psde(*list(cfg['alpha_band_freq'].values()),
                                          sampling_frequency=cfg['sampling_frequency'],
                                          n_jobs=cfg['n_jobs'])
    psde_theta = protocol_utils.init_psde(*list(cfg['theta_band_freq'].values()),
                                          sampling_frequency=cfg['sampling_frequency'],
                                          n_jobs=cfg['n_jobs'])
    #----------------------------------------------------------------------
    # Initialize the feedback sounds
    #----------------------------------------------------------------------
    sound_1, sound_2 = protocol_utils.init_feedback_sounds(cfg['music_state_1_path'],
                                                           cfg['music_state_2_path'])


    #----------------------------------------------------------------------
    # Main
    #----------------------------------------------------------------------
    global_timer   = qc.Timer(autoreset=False)
    internal_timer = qc.Timer(autoreset=True)

    if experiment_mode:
        # Init trigger communication
        #tdef = trigger_def(cfg['trigger_file'])
        #trigger = pyLptControl.Trigger(state, cfg['trigger_device'])
        #if trigger.init(50) == False:
        #    logger.error('\n** Error connecting to trigger device.')
        #    raise RuntimeError

        # Preload the starting voice
        pgmixer.init()
        pgmixer.music.load(cfg['start_voice_file'])

        # Init feedback
        viz = BarVisual(False,
                        screen_pos=cfg['screen_pos'],
                        screen_size=cfg['screen_size'])

        viz.fill()
        viz.put_text('Close your eyes and relax')
        viz.update()

        # Wait a key press
        key = 0xFF & cv2.waitKey(0)
        if key == KEYS['esc'] or not state.value:
            sys.exit(-1)

        #trigger.signal(tdef.INIT)

    inc = 0
    state = 'RATIO_FEEDBACK'

    sound_1.play()
    sound_2.play()

    current_max = 0
    last_ts = None

    #psd_ratio = np.zeros()

    last_ratio = None
    measured_psd_ratios = np.full(cfg['window_size_psd_max'], np.nan)

    recordings = []

    while global_timer.sec() < cfg['global_time']:

        #----------------------------------------------------------------------
        # Data acquisition
        #----------------------------------------------------------------------
        #  Pz = 8
        sr.acquire()
        window, tslist = sr.get_window()    # window = [samples x channels]
        window = window.T                   # window = [channels x samples]

        raw_signal = window[:, -1]

        # Check if proper real-time acquisition
        if last_ts is not None:
            tsnew = np.where(np.array(tslist) > last_ts)[0]
            if len(tsnew) == 0:
                logger.warning('There seems to be delay in receiving data.')
                time.sleep(1)
                continue


        # Spatial filtering
        window = pu.preprocess(window,
                               sfreq=sfreq,
                               spatial=cfg.get('spatial_filter'),
                               spatial_ch=cfg.get('spatial_channels'))

        #----------------------------------------------------------------------
        # Computing the Power Spectrum Densities using multitapers
        #----------------------------------------------------------------------
        # PSD
        #psd_alpha = protocol_utils.compute_psd(window, psde_alpha)
        psd_theta = protocol_utils.compute_psd(window, psde_theta)

        # Ratio Alpha / Theta
        #psd_ratio = psd_alpha / psd_theta
        psd_ratio = psd_theta

        measured_psd_ratios = add_to_queue(measured_psd_ratios, psd_ratio)

        current_music_ratio  = psd_ratio / np.max(measured_psd_ratios[~np.isnan(measured_psd_ratios)])

        if last_ratio is not None:
            applied_music_ratio = last_ratio + (current_music_ratio - last_ratio) * 0.1
        else:
            applied_music_ratio = current_music_ratio

        mix_sounds(style=cfg['music_mix_style'],
                   sounds=(sound_1, sound_2),
                   feature_value=applied_music_ratio)

        recordings += [(raw_signal, psd_ratio, applied_music_ratio)]

        print((f"Ratio Alpha/Theta: {psd_ratio:0.3f}"
               f", current_music_ratio: {current_music_ratio:0.3f}"
               f", applied music ratio: {applied_music_ratio:0.3f}"))

        last_ratio = applied_music_ratio

        last_ts = tslist[-1]
        internal_timer.sleep_atleast(cfg['timer_sleep'])

    sound_1.set_volume(0)
    sound_2.set_volume(0)


    if experiment_mode:
        #trigger.signal(tdef.END)

        # Remove the text
        viz.fill()
        viz.put_text('Recording is finished')
        viz.update()

        # Ending voice
        pgmixer.music.load(cfg['end_voice_file'])
        pgmixer.music.play()
        time.sleep(5)

        # Close cv2 window
        viz.finish()

    with open(f"./online_recording_{time_of_recording}.pkl", 'wb') as f:
        pickle.dump(recordings, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_path = input('Config path? ')
    else:
        cfg_path = sys.argv[1]

    cfg = ProtocolConfig(cfg_path)

    run(cfg)
