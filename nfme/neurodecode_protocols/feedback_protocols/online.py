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
from threading import Thread

import pygame.mixer as pgmixer

import neurodecode.utils.pycnbi_utils as pu



from neurodecode import logger
from neurodecode.utils import q_common as qc
from neurodecode.gui.streams import redirect_stdout_to_queue
from neurodecode.stream_receiver.stream_receiver import StreamReceiver
from neurodecode.protocols.viz_bars import BarVisual
from neurodecode.triggers.trigger_def import trigger_def
import neurodecode.stream_recorder.stream_recorder as recorder
import neurodecode.triggers.pyLptControl as pyLptControl

from nfme.neurodecode_protocols import utils as protocol_utils
from nfme.utils.protocol_config import ProtocolConfig, FeatureType
from nfme.neurodecode_protocols.music import mix_sounds, MusicMixStyle
from nfme.neurodecode_protocols.keyboard_keys import KEYS

os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper
mne.set_log_level('ERROR')          # DEBUG, INFO, WARNING, ERROR, or CRITICAL


def add_to_queue(xs, x):
    xs[:-1] = xs[1:]
    xs[-1] = x
    return xs

def apply_sigmoid(x):
    # scaling the value first to get the full range of the sigmoid fn
    return 1 / (1 + np.exp(-1. * ((x - 0.5 - 0.1) * 10.)))


def run(cfg, amp_name, amp_serial, state=mp.Value('i', 1), experiment_mode=True, baseline=False):
    """
    Online protocol for Alpha/Theta neurofeedback.
    """
    #----------------------------------------------------------------------
    # LSL stream connection
    #----------------------------------------------------------------------

    sr = protocol_utils.connect_lsl_stream(amp_name=amp_name,
                                           amp_serial=amp_serial,
                                           window_size=cfg['window_size'],
                                           buffer_size=cfg['buffer_size'])

    sfreq = sr.get_sample_rate()
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

    pgmixer.init()
    if experiment_mode:
        # Init trigger communication
        trigger_signals = trigger_def(cfg['trigger_file'])
        trigger = pyLptControl.Trigger(state, cfg['trigger_device'])
        if trigger.init(50) == False:
            logger.error('\n** Error connecting to trigger device.')
            raise RuntimeError


        # Preload the starting voice
        print(cfg['start_voice_file'])
        pgmixer.music.load(cfg['start_voice_file'])

        # Init feedback
        viz = BarVisual(False,
                        screen_pos=cfg['screen_pos'],
                        screen_size=cfg['screen_size'])

        viz.fill()
        viz.put_text('Close your eyes and relax')
        viz.update()

        pgmixer.music.play()

        # Wait a key press
        key = 0xFF & cv2.waitKey(0)
        if key == KEYS['esc'] or not state.value:
            sys.exit(-1)

        print('recording started')

        trigger.signal(trigger_signals.INIT)

    state = 'RATIO_FEEDBACK'

    if not baseline:
        sound_1.play(loops=-1)
        sound_2.play(loops=-1)

    current_max = 0
    last_ts = None

    last_ratio = None
    measured_psd_ratios = np.full(cfg['window_size_psd_max'], np.nan)

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
        if not baseline:
            if cfg['feature_type'] == FeatureType.THETA:
                psd_theta = protocol_utils.compute_psd(window, psde_theta)

                feature = psd_theta
            elif cfg['feature_type'] == FeatureType.ALPHA_THETA:
                psd_alpha = protocol_utils.compute_psd(window, psde_alpha)
                psd_theta = protocol_utils.compute_psd(window, psde_theta)

                feature = psd_alpha / psd_theta

            measured_psd_ratios = add_to_queue(measured_psd_ratios, feature)
            current_music_ratio  = feature / np.max(measured_psd_ratios[~np.isnan(measured_psd_ratios)])

            #current_music_ratio  = feature / np.max(measured_psd_ratios)

            if last_ratio is not None:
                applied_music_ratio = last_ratio + (current_music_ratio - last_ratio) * 0.25
            else:
                applied_music_ratio = current_music_ratio

            mix_sounds(style=cfg['music_mix_style'],
                       sounds=(sound_1, sound_2),
                       feature_value=applied_music_ratio)

            print((f"{cfg['feature_type']}: {feature:0.3f}"
                   f"\t, current_music_ratio: {current_music_ratio:0.3f}"
                   f"\t, applied music ratio: {applied_music_ratio:0.3f}"
                   ))

            last_ratio = applied_music_ratio


        last_ts = tslist[-1]
        internal_timer.sleep_atleast(cfg['timer_sleep'])


    if not baseline:
        sound_1.fadeout(3)
        sound_2.fadeout(3)


    if experiment_mode:
        trigger.signal(trigger_signals.END)

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

    print('done')

def launching_subprocesses(*args):
    """
    Launch subprocesses

    processesToLaunch = list of tuple containing the functions to launch
    and their args
    """
    launchedProcesses = dict()

    for p in args[1:]:
        launchedProcesses[p[0]] = mp.Process(target=p[1], args=p[2])
        launchedProcesses[p[0]].start()

    # Wait that the protocol is finished to stop recording
    launchedProcesses['protocol'].join()

    try:
        launchedProcesses['recording']
        recordState = args[1][2][0]     #  Sharing variable
        with recordState.get_lock():
            recordState.value = 0
    except:
        pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('please provide the path for the config to load')
    else:
        cfg_path = sys.argv[1]

    cfg = ProtocolConfig(cfg_path)

    #  0=stop, 1=start, 2=wait
    record_state = mp.Value('i', 2)
    protocol_state = mp.Value('i', 1)
    record_dir = '.'

    amp_name, amp_serial = protocol_utils.find_lsl_stream(mp.Value('i', 1),
                                                          amp_name=cfg.get('amp_name', None),
                                                          amp_serial=cfg.get('amp_serial', None))

    processesToLaunch = [('recording', recorder.run_gui, [record_state, protocol_state, record_dir,
                                                          None, amp_name, amp_serial, False]), \
                         ('protocol', run, [cfg, amp_name, amp_serial, protocol_state])]

    launchedProcess = Thread(target=launching_subprocesses, args=processesToLaunch)
    launchedProcess.start()

    #run(cfg)
