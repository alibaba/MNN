#!/usr/bin/env python3
"""Demonstrate MNN audio windows and WAV feature extraction."""

import argparse

import MNN.audio as audio


def demo_window_functions():
    """Generate Hamming and Hann windows."""
    hamming = audio.hamming_window(400, periodic=True)
    hann = audio.hanning_window(400, periodic=True)
    print(f"Hamming window shape: {hamming.shape}")
    print(f"Hann window shape: {hann.shape}")


def demo_audio_features(filename, output_filename=None):
    """Load a WAV file at 16 kHz and extract speech features."""
    waveform, sample_rate = audio.load(filename, sr=16000)
    if waveform is None or sample_rate <= 0:
        raise RuntimeError(f"Could not load WAV file: {filename}")
    print(f"Loaded waveform: {waveform.shape}, {sample_rate} Hz")

    spectrogram = audio.spectrogram(
        waveform, n_fft=400, hop_length=160, win_length=400,
        window_type=audio.HANNING, center=True, pad_mode=audio.REFLECT,
    )
    print(f"Spectrogram shape: {spectrogram.shape}")

    mel = audio.mel_spectrogram(waveform, 80, 400, sample_rate, hop_length=160)
    print(f"Mel-spectrogram shape: {mel.shape}")

    fbank = audio.fbank(waveform, sample_rate=sample_rate, n_mels=80)
    print(f"Fbank shape: {fbank.shape}")

    whisper = audio.whisper_fbank(waveform, sample_rate=sample_rate, n_mels=128)
    print(f"Whisper Fbank shape: {whisper.shape}")

    if output_filename:
        if not audio.save(output_filename, waveform, sample_rate):
            raise RuntimeError(f"Could not save WAV file: {output_filename}")
        print(f"Saved resampled waveform: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", nargs="?", help="Input PCM WAV file")
    parser.add_argument("--output", help="Save the resampled waveform as a WAV file")
    arguments = parser.parse_args()
    if arguments.output and not arguments.audio_file:
        parser.error("--output requires an input WAV file")

    demo_window_functions()
    if arguments.audio_file:
        demo_audio_features(arguments.audio_file, arguments.output)
    else:
        print("Pass a WAV file to run audio I/O and feature extraction.")


if __name__ == "__main__":
    main()
