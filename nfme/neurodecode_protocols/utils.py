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

def find_lsl_stream(state, amp_name=None, amp_serial=None):
    """
    Find the amplifier name and its serial number to connect to

    cfg = config file
    state = GUI sharing variable
    """
    if any(v is None for v in (amp_name, amp_serial)):
        amp_name, amp_serial = pu.search_lsl(state, ignore_markers=True)

    return amp_name, amp_serial


def connect_lsl_stream(amp_name, amp_serial, window_size, buffer_size):
    """
    Connect to the lsl stream corresponding to the provided amplifier
    name and serial number

    amp_name =  amplifier's name to connect to
    amp_serial = amplifier's serial number
    """
    # Open the stream
    return StreamReceiver(window_size=window_size,
                        buffer_size=buffer_size,
                        amp_serial=amp_serial,
                        eeg_only=False,
                        amp_name=amp_name)


def init_psde(band_lower_bound, band_upper_bound, sampling_frequency, n_jobs=1):
    """
    Initialize the PSD estimators (MNE lib) for computation of alpha
    and theta bands PSD

    """

    return mne.decoding.PSDEstimator(sfreq=sampling_frequency,
                                     fmin=band_lower_bound,
                                     fmax=band_upper_bound,
                                     bandwidth=None,
                                     adaptive=False,
                                     low_bias=True,
                                     n_jobs=n_jobs,
                                     normalization='length',
                                     verbose=None)


def init_feedback_sounds(path1, path2):
    """
    Initialize the sounds for alpha and theta feedbacks
    """
    pgmixer.init()
    pgmixer.set_num_channels(4)

    m1 = pgmixer.Sound(path1)
    m2 = pgmixer.Sound(path2)
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
