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

import utils as protocol_utils

os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper
mne.set_log_level('ERROR')          # DEBUG, INFO, WARNING, ERROR, or CRITICAL

def add_to_queue(xs, x):
    xs[:-1] = xs[1:]
    xs[-1] = x
    return xs

def biomarker_to_music():
    pass


#----------------------------------------------------------------------
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
    amp_name, amp_serial = protocol_utils.find_lsl_stream(cfg, state)

    # Connect to lsl stream
    sr = protocol_utils.connect_lsl_stream(cfg, amp_name, amp_serial)

    # Get sampling rate
    sfreq = sr.get_sample_rate()

    # Get trigger channel
    trg_ch = sr.get_trigger_channel()

    #----------------------------------------------------------------------
    # PSD estimators initialization
    #----------------------------------------------------------------------
    psde_alpha, psde_theta = protocol_utils.init_psde(cfg, sfreq)

    #----------------------------------------------------------------------
    # Initialize the feedback sounds
    #----------------------------------------------------------------------
    sound_1, sound_2 = protocol_utils.init_feedback_sounds(cfg)

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

    while global_timer.sec() < cfg.GLOBAL_TIME:

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
        window = pu.preprocess(window, sfreq=sfreq, spatial=cfg.SPATIAL_FILTER, spatial_ch=cfg.SPATIAL_CHANNELS)

        #----------------------------------------------------------------------
        # Computing the Power Spectrum Densities using multitapers
        #----------------------------------------------------------------------
        # PSD
        psd_alpha = protocol_utils.compute_psd(window, psde_alpha)#, cfg.ALPHA_REF)
        psd_theta = protocol_utils.compute_psd(window, psde_theta)#, cfg.THETA_REF)

        # Ratio Alpha / Theta
        psd_ratio = psd_alpha / psd_theta

        if psd_ratio > current_max:
            current_max = psd_ratio

        music_ratio = psd_ratio / current_max

        sound_1.set_volume(music_ratio)
        sound_2.set_volume(1 - music_ratio)


        #----------------------------------------------------------------------
        # Auditory feedback
        #----------------------------------------------------------------------
        #if state == 'RATIO_FEEDBACK':
        #    if psd_ratio > 1:
        #        theta_sound.stop(fade_ms=1000)
        #        alpha_sound.play(fade_ms=500)
        #    else:
        #        alpha_sound.stop(fade_ms=1000)
        #        theta_sound.start(fade_ms=500)
        #    state = 'SUPRA_FEEDBACK'

        #elif state == 'SUPRA_FEEDBACK':
        #    if psd_alpha > cfg.ALPHA_THR:
        #        alpha_sup_sound.play()
        #    if psd_theta > cfg.THETA_THR:
        #        theta_sup_sound.play()

        #    inc = inc + 1
        #    if inc == 4:
        #        inc = 0
        #        state = 'RATIO_FEEDBACK'

        print(f"Ratio Alpha/Theta: {psd_ratio:0.3f}, music_ratio: {music_ratio:0.3f}")

        last_ts = tslist[-1]
        internal_timer.sleep_atleast(cfg.TIMER_SLEEP)

#----------------------------------------------------------------------
def batch_run(cfg_module):
    """
    For batch script
    """
    cfg = pu.load_config(cfg_module)
    check_config(cfg)
    run(cfg)

#----------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_module = input('Config module name? ')
    else:
        cfg_module = sys.argv[1]

    #cfg_module = '/home/sam/proj/epfl/eeg-meditation/new_scripts/AlphaTheta/sam-AlphaTheta/config_online_sam-AlphaTheta'
    batch_run(cfg_module)
