# sherpa-mnn/python/tests/test_keyword_spotter.py
#
# Copyright (c)  2024  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_keyword_spotter_py

import unittest
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_mnn

d = "/tmp/onnx-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
# to download pre-trained models for testing


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


class TestKeywordSpotter(unittest.TestCase):
    def test_zipformer_transducer_en(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
                decoder = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                joiner = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
            else:
                encoder = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                decoder = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                joiner = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"

            tokens = (
                f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/tokens.txt"
            )
            keywords_file = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt"
            wave0 = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/0.wav"
            wave1 = f"{d}/sherpa-mnn-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav"

            if not Path(encoder).is_file():
                print("skipping test_zipformer_transducer_en()")
                return
            keyword_spotter = sherpa_mnn.KeywordSpotter(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=1,
                keywords_file=keywords_file,
                provider="cpu",
            )
            streams = []
            waves = [wave0, wave1]
            for wave in waves:
                s = keyword_spotter.create_stream()
                samples, sample_rate = read_wave(wave)
                s.accept_waveform(sample_rate, samples)

                tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                s.accept_waveform(sample_rate, tail_paddings)
                s.input_finished()
                streams.append(s)

            results = [""] * len(streams)
            while True:
                ready_list = []
                for i, s in enumerate(streams):
                    if keyword_spotter.is_ready(s):
                        ready_list.append(s)
                    r = keyword_spotter.get_result(s)
                    if r:
                        print(f"{r} is detected.")
                        results[i] += f"{r}/"

                        keyword_spotter.reset_stream(s)

                if len(ready_list) == 0:
                    break
                keyword_spotter.decode_streams(ready_list)
            for wave_filename, result in zip(waves, results):
                print(f"{wave_filename}\n{result[0:-1]}")
                print("-" * 10)

    def test_zipformer_transducer_cn(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
                decoder = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                joiner = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
            else:
                encoder = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                decoder = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
                joiner = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"

            tokens = (
                f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt"
            )
            keywords_file = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt"
            wave0 = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav"
            wave1 = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/4.wav"
            wave2 = f"{d}/sherpa-mnn-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/5.wav"

            if not Path(encoder).is_file():
                print("skipping test_zipformer_transducer_cn()")
                return
            keyword_spotter = sherpa_mnn.KeywordSpotter(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=1,
                keywords_file=keywords_file,
                provider="cpu",
            )
            streams = []
            waves = [wave0, wave1, wave2]
            for wave in waves:
                s = keyword_spotter.create_stream()
                samples, sample_rate = read_wave(wave)
                s.accept_waveform(sample_rate, samples)

                tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                s.accept_waveform(sample_rate, tail_paddings)
                s.input_finished()
                streams.append(s)

            results = [""] * len(streams)
            while True:
                ready_list = []
                for i, s in enumerate(streams):
                    if keyword_spotter.is_ready(s):
                        ready_list.append(s)
                    r = keyword_spotter.get_result(s)
                    if r:
                        print(f"{r} is detected.")
                        results[i] += f"{r}/"

                        keyword_spotter.reset_stream(s)

                if len(ready_list) == 0:
                    break
                keyword_spotter.decode_streams(ready_list)
            for wave_filename, result in zip(waves, results):
                print(f"{wave_filename}\n{result[0:-1]}")
                print("-" * 10)


if __name__ == "__main__":
    unittest.main()
