#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-onnx Python API to do keyword spotting
from wave file(s).

Please refer to
https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
to download pre-trained models.
"""
import argparse
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sherpa_mnn


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


def create_keyword_spotter():
    kws = sherpa_mnn.KeywordSpotter(
        tokens="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
        encoder="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        decoder="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        joiner="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
        num_threads=2,
        keywords_file="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt",
        provider="cpu",
    )

    return kws


def main():
    kws = create_keyword_spotter()

    wave_filename = (
        "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav"
    )

    samples, sample_rate = read_wave(wave_filename)

    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)

    print("----------Use pre-defined keywords----------")
    s = kws.create_stream()
    s.accept_waveform(sample_rate, samples)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()
    while kws.is_ready(s):
        kws.decode_stream(s)
        r = kws.get_result(s)
        if r != "":
            # Remember to call reset right after detected a keyword
            kws.reset_stream(s)

            print(f"Detected {r}")

    print("----------Use pre-defined keywords + add a new keyword----------")

    s = kws.create_stream("y ǎn y uán @演员")
    s.accept_waveform(sample_rate, samples)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()
    while kws.is_ready(s):
        kws.decode_stream(s)
        r = kws.get_result(s)
        if r != "":
            # Remember to call reset right after detected a keyword
            kws.reset_stream(s)

            print(f"Detected {r}")

    print("----------Use pre-defined keywords + add 2 new keywords----------")

    s = kws.create_stream("y ǎn y uán @演员/zh ī m íng @知名")
    s.accept_waveform(sample_rate, samples)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()
    while kws.is_ready(s):
        kws.decode_stream(s)
        r = kws.get_result(s)
        if r != "":
            # Remember to call reset right after detected a keyword
            kws.reset_stream(s)

            print(f"Detected {r}")


if __name__ == "__main__":
    main()
