#!/usr/bin/env python3

"""
This script shows how to use Python APIs for spoken languge identification.
It detects the language spoken in the given wave file.

Usage:

1. Download a whisper multilingual model. We use a tiny model below.
Please refer to https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
to download more models.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
rm sherpa-onnx-whisper-tiny.tar.bz2

We only use the int8.onnx models below.

2. Download a test wave.

You can find many wave files for different languages at
https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/tree/main/test_wavs

wget https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/resolve/main/test_wavs/de-german.wav

python3 ./python-api-examples/spoken-language-identification.py
  --whisper-encoder=sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx \
  --whisper-decoder=sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx \
  --num-threads=1 \
  ./de-german.wav
"""

import argparse
import logging
import time
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_mnn


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--whisper-encoder",
        required=True,
        type=str,
        help="Path to a multilingual whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        required=True,
        type=str,
        help="Path to a multilingual whisper decoder model",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="The input sound file to identify. It must be of WAVE"
        "format with a single channel, and each sample has 16-bit, "
        "i.e., int16_t. "
        "The sample rate of the file can be arbitrary and does not need to "
        "be 16 kHz",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html to download it"
    )


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def main():
    args = get_args()
    assert_file_exists(args.whisper_encoder)
    assert_file_exists(args.whisper_decoder)
    assert args.num_threads > 0, args.num_threads
    config = sherpa_mnn.SpokenLanguageIdentificationConfig(
        whisper=sherpa_mnn.SpokenLanguageIdentificationWhisperConfig(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
        ),
        num_threads=args.num_threads,
        debug=args.debug,
        provider=args.provider,
    )
    slid = sherpa_mnn.SpokenLanguageIdentification(config)

    samples, sample_rate = read_wave(args.sound_file)

    start_time = time.time()
    stream = slid.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    lang = slid.compute(stream)
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    audio_duration = len(samples) / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    logging.info(f"File: {args.sound_file}")
    logging.info(f"Detected language: {lang}")
    logging.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
    logging.info(f"Audio duration in seconds: {audio_duration:.3f}")
    logging.info(
        f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
