#!/usr/bin/env python3
"""
MNN Audio Module Demo

This script demonstrates basic audio processing operations using MNN.audio:
- Loading and saving audio files
- Computing spectrograms and mel-spectrograms
- Extracting Fbank features (commonly used in speech recognition)
"""

import MNN
import MNN.audio as audio

def demo_load_save():
    """Demo: Load and save audio files"""
    print("=== Audio Load/Save Demo ===")

    # Load audio file (supports common formats: wav, mp3, flac, etc.)
    # audio.load(filename, sr, frame_offset, num_frames) - pymnn/src/audio.h
    waveform, sample_rate = audio.load("input.wav", 0, 0, -1)
    print(f"Loaded audio: sample_rate={sample_rate} Hz, shape={waveform.shape}")

    # Save audio to file
    audio.save("output.wav", waveform, sample_rate)
    print("Saved audio to output.wav")

def demo_spectrogram():
    """Demo: Compute spectrogram from audio"""
    print("\n=== Spectrogram Demo ===")

    # Load audio
    # audio.load(filename, sr, frame_offset, num_frames) - pymnn/src/audio.h
    waveform, sr = audio.load("input.wav", 0, 0, -1)

    # Compute spectrogram with positional parameters
    # audio.spectrogram(waveform, n_fft, hop_length, win_length, window_type,
    #                   pad_left, pad_right, center, normalized, pad_mode, power)
    # window_type is int (0=hann, 1=hamming, etc. - check audio.h for enum values)
    spec = audio.spectrogram(waveform, 400, 160, 400, 0, 0, 0, 1, 0, 0, 2.0)
    print(f"Spectrogram shape: {spec.shape}")

def demo_mel_spectrogram():
    """Demo: Compute mel-spectrogram"""
    print("\n=== Mel-Spectrogram Demo ===")

    # Load audio with resampling to 16kHz
    # audio.load(filename, sr, frame_offset, num_frames) - pymnn/src/audio.h
    waveform, sr = audio.load("input.wav", 16000, 0, -1)

    # Compute mel-spectrogram with positional parameters
    # WARNING: The binding only accepts 4 int parameters due to format string bug
    # audio.mel_spectrogram(waveform, n_mels, n_fft, sample_rate) - pymnn/src/audio.h
    mel_spec = audio.mel_spectrogram(waveform, 80, 400, 16000)
    print(f"Mel-spectrogram shape: {mel_spec.shape}")

def demo_fbank():
    """Demo: Extract Fbank features (Kaldi-style)"""
    print("\n=== Fbank Features Demo ===")

    # Load audio
    # audio.load(filename, sr, frame_offset, num_frames) - pymnn/src/audio.h
    waveform, sr = audio.load("input.wav", 16000, 0, -1)

    # Compute Fbank features (commonly used in ASR)
    # audio.fbank(waveform, sample_rate, n_mels, n_fft, hop_length, dither, preemphasis)
    fbank_features = audio.fbank(waveform, 16000, 80, 400, 160, 0.0, 0.97)
    print(f"Fbank features shape: {fbank_features.shape}")

def demo_whisper_fbank():
    """Demo: Whisper-style Fbank features"""
    print("\n=== Whisper Fbank Demo ===")

    # Load audio
    # audio.load(filename, sr, frame_offset, num_frames) - pymnn/src/audio.h
    waveform, sr = audio.load("input.wav", 16000, 0, -1)

    # Compute Whisper-compatible Fbank features
    # audio.whisper_fbank(waveform, sample_rate, n_mels, n_fft, hop_length, chunk_len)
    whisper_fbank = audio.whisper_fbank(waveform, 16000, 128, 400, 160, 30)
    print(f"Whisper Fbank shape: {whisper_fbank.shape}")

def demo_window_functions():
    """Demo: Window functions for audio processing"""
    print("\n=== Window Functions Demo ===")

    window_size = 400

    # Hamming window
    # audio.hamming_window(window_size, periodic, alpha, beta) - pymnn/src/audio.h
    hamming = audio.hamming_window(window_size, 1, 0.54, 0.46)
    print(f"Hamming window shape: {hamming.shape}")

    # Hann window (note: real function name is hann_window, not hanning_window)
    # audio.hann_window(window_size, periodic) - pymnn/src/audio.h
    hann = audio.hann_window(window_size, 1)
    print(f"Hann window shape: {hann.shape}")

if __name__ == "__main__":
    print("MNN Audio Module Demo\n")
    print("Note: Replace 'input.wav' with your actual audio file path")
    print("="*50)

    # Run demos (comment out if you don't have input.wav)
    # demo_load_save()
    # demo_spectrogram()
    # demo_mel_spectrogram()
    # demo_fbank()
    # demo_whisper_fbank()
    demo_window_functions()

    print("\n" + "="*50)
    print("Demo completed!")
