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

#----------------------------------------------------------------------
def check_config(cfg, critical_vars, optional_vars):
    """
    Ensure that the config file contains the parameters
    """
    critical_vars = {
        'COMMON': ['ALPHA_CHANNELS',
                   'THETA_CHANNELS',
                   'ALPHA_THR',
                   'THETA_THR',
                   'ALPHA_REF',
                   'THETA_REF',
                   'WINDOWSIZE',
                   'MUSIC_STATE_1_PATH',
                   'MUSIC_STATE_2_PATH',
                   'STREAMBUFFER'],
        'ALPHA_FREQ': ['min', 'max'],
        'THETA_FREQ': ['min', 'max'],
        }

    optional_vars = {
        'AMP_NAME':None,
        'AMP_SERIAL':None,
        'SPATIAL_FILTER': None,
        'SPATIAL_CHANNELS': None,
        'GLOBAL_TIME': 30.0 * 60,
        'NJOBS': 1,
    }

    for key in critical_vars['COMMON']:
        if not hasattr(cfg, key):
            logger.error('%s is a required parameter' % key)
            raise RuntimeError

    if not hasattr(cfg, 'ALPHA_FREQ'):
        logger.error('"ALPHA_FREQ" not defined in config.')
        raise RuntimeError
    for v in critical_vars['ALPHA_FREQ']:
        if v not in cfg.ALPHA_FREQ:
            logger.error('%s not defined in config.' % v)
            raise RuntimeError

    if not hasattr(cfg, 'THETA_FREQ'):
        logger.error('"THETA_FREQ" not defined in config.')
        raise RuntimeError
    for v in critical_vars['THETA_FREQ']:
        if v not in cfg.THETA_FREQ:
            logger.error('%s not defined in config.' % v)
            raise RuntimeError

    for key in optional_vars:
        if not hasattr(cfg, key):
            setattr(cfg, key, optional_vars[key])
            logger.warning('Setting undefined parameter %s=%s' % (key, getattr(cfg, key)))


#----------------------------------------------------------------------
def find_lsl_stream(cfg, state):
    """
    Find the amplifier name and its serial number to connect to

    cfg = config file
    state = GUI sharing variable
    """
    if any((not hasattr(cfg, key) for key in ['AMP_NAME', 'AMP_SERIAL'])):
        amp_name, amp_serial = pu.search_lsl(state, ignore_markers=True)
    else:
        amp_name = cfg.AMP_NAME
        amp_serial = cfg.AMP_SERIAL

    return amp_name, amp_serial

#----------------------------------------------------------------------
def connect_lsl_stream(cfg, amp_name, amp_serial):
    """
    Connect to the lsl stream corresponding to the provided amplifier
    name and serial number

    cfg = config file
    amp_name =  amplifier's name to connect to
    amp_serial = amplifier's serial number
    """
    # Open the stream
    sr = StreamReceiver(window_size=cfg.WINDOWSIZE, buffer_size=cfg.STREAMBUFFER, amp_serial=amp_serial, eeg_only=False, amp_name=amp_name)

    return sr

#----------------------------------------------------------------------
def init_psde(cfg, sfreq):
    """
    Initialize the PSD estimators (MNE lib) for computation of alpha
    and theta bands PSD

    cfg = config file
    sfreq = sampling rate
    """
    psde_alpha = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=cfg.ALPHA_FREQ['min'], fmax=cfg.ALPHA_FREQ['max'], bandwidth=None, \
             adaptive=False, low_bias=True, n_jobs=cfg.NJOBS, normalization='length', verbose=None)

    psde_theta = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=cfg.THETA_FREQ['min'], fmax=cfg.THETA_FREQ['max'], bandwidth=None, \
             adaptive=False, low_bias=True, n_jobs=cfg.NJOBS, normalization='length', verbose=None)

    return psde_alpha, psde_theta

#----------------------------------------------------------------------
def init_feedback_sounds(cfg):
    """
    Initialize the sounds for alpha and theta feedbacks
    """
    pgmixer.init()
    pgmixer.set_num_channels(4)

    m1 = pgmixer.Sound(cfg.MUSIC_STATE_1_PATH)
    m2 = pgmixer.Sound(cfg.MUSIC_STATE_1_PATH)
    m1.set_volume(1.0)
    m2.set_volume(0.0)

    return m1, m2

    # Normal feedbacks
    #alpha_sound = pgmixer.Sound(cfg.ALPHA_FB_PATH)
    #theta_sound = pgmixer.Sound(cfg.THETA_FB_PATH)
    #alpha_sound.set_volume(1.0)
    #theta_sound.set_volume(1.0)

    ## Suprethreshold feedbacks
    #alpha_sup_sound = pgmixer.Sound(cfg.ALPHA_SUP_FB_PATH)
    #theta_sup_sound = pgmixer.Sound(cfg.THETA_SUP_FB_PATH)
    #alpha_sup_sound.set_volume(1.0)
    #theta_sup_sound.set_volume(1.0)

    #return alpha_sound, theta_sound, alpha_sup_sound, theta_sup_sound

#----------------------------------------------------------------------
def compute_psd(window, psde, psd_ref=None):
    """
    Compute the relative PSD

    psde = PSD estimator
    psd_ref = psd reference value defined during an offline run
    """
    psd = psde.transform(window.reshape((1, window.shape[0], -1)))
    psd = psd.reshape((psd.shape[1], psd.shape[2]))                 # channels x frequencies
    psd =  np.sum(psd, axis=1)                                      #  Over frequencies
    m_psd = np.mean(psd)                                            #  Over channels


    if psd_ref is None:
        return m_psd
    else:
        return m_psd / psd_ref
