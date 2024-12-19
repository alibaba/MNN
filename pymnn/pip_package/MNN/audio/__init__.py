from _mnncengine.audio import *
import _mnncengine.audio as _F
import MNN.expr as _expr
import MNN.numpy as _np
import MNN

# Enum Types
# enum WINDOW_TYPE
HAMMING = 0
HANNING = 1
POVEY = 2
RECTANGULAR = 3
BLACKMAN = 4
# enum PadValueMode
CONSTANT = 0
REFLECT = 1
SYMMETRIC = 2
EDGE = 3

"""
Loads a portion of an audio file.

Parameters:
    filename (str): The path to the audio file.
    frame_offset (int): The offset in frames from which to start loading the audio data. Default is 0.
    num_frames (int): The number of frames to load. If set to -1, the entire audio file will be loaded. Default is -1.

Returns:
    The result of loading the specified portion of the audio var and the sample rate.
"""
def load(filename, sr = 0, frame_offset = 0, num_frames = -1):
    return _F.load(filename, sr, frame_offset, num_frames)

"""
Saves an audio var to a file.
Parameters:
    filename (str): The path to the audio file.
    audio (Var): The audio var to save.
    sample_rate (int): The sample rate of the audio var.
Returns:
    None
"""
def save(filename, audio, sample_rate):
    return _F.save(filename, audio, sample_rate)

"""
Generates a Hamming window.
Parameters:
    window_size (int): The size of the window.
    periodic (bool): Whether the window is periodic. Default is False.
    alpha (float): The alpha parameter of the Hamming window. Default is 0.54.
    beta (float): The beta parameter of the Hamming window. Default is 0.46.
Returns:
    The Hamming window.
"""
def hamming_window(window_size, periodic = False, alpha = 0.54, beta = 0.46):
    return _F.hamming_window(window_size, periodic, alpha, beta)

"""
Generates a Hann window.
Parameters:
    window_size (int): The size of the window.
    periodic (bool): Whether the window is periodic. Default is False.
Returns:
    The Hann window.
"""
def hanning_window(window_size, periodic = False):
    return _F.hanning_window(window_size, periodic)

def melscale_fbanks(n_mels, n_fft, sampe_rate = 16000, htk = True, norm = False,
                    f_min = 0.0, f_max = 0.0):
    return _F.melscale_fbanks(n_mels, n_fft, sampe_rate, htk, norm, f_min, f_max)

def spectrogram(waveform, n_fft = 400, hop_length = 0, win_length = 0, window_type = HANNING,
                pad_left = 0, pad_right = 0, center = False, normalized = False, pad_mode = REFLECT,
                power = 2.0):
    return _F.spectrogram(waveform, n_fft, hop_length, win_length, window_type, pad_left,
                          pad_right, center, normalized, pad_mode, power)


def mel_spectrogram(waveform, n_mels, n_fft, sampe_rate = 16000, htk = True, norm = False,
                    f_min = 0.0, f_max = 0.0, hop_length = 0, win_length = 0, window_type = HANNING,
                    pad_left = 0, pad_right = 0, center = False, normalized = False, pad_mode = REFLECT,
                    power = 2.0):
    return _F.mel_spectrogram(waveform, n_mels, n_fft, sampe_rate, htk, norm, f_min, f_max,
                              hop_length, win_length, window_type, pad_left, pad_right, center,
                              normalized, pad_mode, power)

def fbank(waveform, sample_rate = 16000, n_mels = 80, n_fft = 400, hop_length = 160,
          dither = 0.0, preemphasis = 0.97):
    return _F.fbank(waveform, sample_rate, n_mels, n_fft, hop_length, dither, preemphasis)


def whisper_fbank(waveform, sample_rate = 16000, n_mels = 128, n_fft = 400,
                  hop_length = 160, chunk_len = 30):
    return _F.whisper_fbank(waveform, sample_rate, n_mels, n_fft, hop_length, chunk_len)