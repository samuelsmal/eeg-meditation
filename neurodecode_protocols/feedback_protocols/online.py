#!/usr/bin/env python3
#coding:utf-8

import mne
import os
import sys

import time
import numpy as np
import multiprocessing as mp
import pygame.mixer as pgmixer

import neurodecode.utils.pycnbi_utils as pu

from neurodecode import logger
from neurodecode.utils import q_common as qc
from neurodecode.gui.streams import redirect_stdout_to_queue
from neurodecode.stream_receiver.stream_receiver import StreamReceiver

from lib import utils as protocol_utils
from lib.config import ProtocolConfig

os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper
mne.set_log_level('ERROR')          # DEBUG, INFO, WARNING, ERROR, or CRITICAL

def add_to_queue(xs, x):
    xs[:-1] = xs[1:]
    xs[-1] = x
    return xs


def run(cfg, state=mp.Value('i', 1), queue=None):
    """
    Online protocol for Alpha/Theta neurofeedback.
    """
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
                                          sampling_frequency=cfg['sampling_frequency'])
    psde_theta = protocol_utils.init_psde(*list(cfg['theta_band_freq'].values()),
                                          sampling_frequency=cfg['sampling_frequency'])

    #----------------------------------------------------------------------
    # Initialize the feedback sounds
    #----------------------------------------------------------------------
    sound_1, sound_2 = protocol_utils.init_feedback_sounds(cfg['music_state_1_path'],
                                                           cfg['music_state_2_path'])


    #----------------------------------------------------------------------
    # Main
    #----------------------------------------------------------------------
    global_timer = qc.Timer(autoreset=False)
    internal_timer = qc.Timer(autoreset=True)

    inc = 0
    state = 'RATIO_FEEDBACK'

    sound_1.play()
    sound_2.play()

    current_max = 0
    last_ts = None

    #psd_ratio = np.zeros()

    while global_timer.sec() < cfg['global_time']:

        #----------------------------------------------------------------------
        # Data acquisition
        #----------------------------------------------------------------------
        #  Pz = 8
        sr.acquire()
        window, tslist = sr.get_window()    # window = [samples x channels]
        window = window.T                   # window = [channels x samples]

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
        psd_alpha = protocol_utils.compute_psd(window, psde_alpha)
        psd_theta = protocol_utils.compute_psd(window, psde_theta)

        # Ratio Alpha / Theta
        psd_ratio = psd_alpha / psd_theta

        if psd_ratio > current_max:
            current_max = psd_ratio

        music_ratio = psd_ratio / current_max

        sound_1.set_volume(music_ratio)
        sound_2.set_volume(1 - music_ratio)


        print(f"Ratio Alpha/Theta: {psd_ratio:0.3f}, music_ratio: {music_ratio:0.3f}")

        last_ts = tslist[-1]
        internal_timer.sleep_atleast(cfg['timer_sleep'])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_path = input('Config path? ')
    else:
        cfg_path = sys.argv[1]

    cfg = ProtocolConfig(cfg_path)

    run(cfg)
