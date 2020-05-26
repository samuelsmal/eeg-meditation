#    NFME NeuroFeedback for Meditation using EEG recordings
#    Copyright (C) 2020 Samuel von Baussnern, samuel.edlervonbaussnern@epfl.ch
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from scipy import signal


def _butter_bandpass_(lowcut_frequency, highcut_frequency, sampling_frequency, order=5):
    """Returns a bandpass filter using second-order sections.

    See here: https://stackoverflow.com/a/48677312
    """
    nyq = 0.5 * sampling_frequency
    l = lowcut_frequency / nyq
    h = highcut_frequency / nyq

    # the data is in digital form -> no analog
    return signal.butter(order, [l, h], analog=False, btype='band', output='sos')


def bandpass_filter(data, band, sampling_frequency, order=5):
    """will return a numpy array"""
    sos = _butter_bandpass_(*band, sampling_frequency, order=order)
    return signal.sosfilt(sos, data)


def apply_bandpass_filter(combined_data, bandpass_filter_bands, sampling_frequency):
    return {recording_type: [rec.apply(lambda xs: bandpass_filter(xs,
                                                                  bandpass_filter_bands,
                                                                  sampling_frequency),
                                       axis=0)
                             for rec in recordings]
                                    for recording_type, recordings in combined_data.items()}


def remove_reference_signal(data, reference_electrode):
    for c in data.columns:
        data.loc[:, c] = data.loc[:, c] - data.loc[:, reference_electrode]

    return data


def apply_remove_reference_signal(combined_data, reference_electrode):
    return {recording_type: [remove_reference_signal(rec, reference_electrode) for rec in recordings]
            for recording_type, recordings in combined_data.items()}
