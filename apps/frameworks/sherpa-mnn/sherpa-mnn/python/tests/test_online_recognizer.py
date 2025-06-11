# sherpa-mnn/python/tests/test_online_recognizer.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_online_recognizer_py

import unittest
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_mnn

d = "/tmp/icefall-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
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


class TestOnlineRecognizer(unittest.TestCase):
    def test_transducer_single_file(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx"
                decoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx"
            else:
                encoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
                decoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx"

            tokens = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
            wave0 = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav"

            if not Path(encoder).is_file():
                print("skipping test_transducer_single_file()")
                return

            for decoding_method in ["greedy_search", "modified_beam_search"]:
                recognizer = sherpa_mnn.OnlineRecognizer.from_transducer(
                    encoder=encoder,
                    decoder=decoder,
                    joiner=joiner,
                    tokens=tokens,
                    num_threads=1,
                    decoding_method=decoding_method,
                    provider="cpu",
                )
                s = recognizer.create_stream()
                samples, sample_rate = read_wave(wave0)
                s.accept_waveform(sample_rate, samples)

                tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                s.accept_waveform(sample_rate, tail_paddings)

                s.input_finished()
                while recognizer.is_ready(s):
                    recognizer.decode_stream(s)
                print(recognizer.get_result(s))

    def test_transducer_multiple_files(self):
        for use_int8 in [True, False]:
            if use_int8:
                encoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx"
                decoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx"
            else:
                encoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
                decoder = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
                joiner = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx"

            tokens = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
            wave0 = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav"
            wave1 = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav"
            wave2 = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/2.wav"
            wave3 = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/3.wav"
            wave4 = f"{d}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/8k.wav"

            if not Path(encoder).is_file():
                print("skipping test_transducer_multiple_files()")
                return

            for decoding_method in ["greedy_search", "modified_beam_search"]:
                recognizer = sherpa_mnn.OnlineRecognizer.from_transducer(
                    encoder=encoder,
                    decoder=decoder,
                    joiner=joiner,
                    tokens=tokens,
                    num_threads=1,
                    decoding_method=decoding_method,
                    provider="cpu",
                )
                streams = []
                waves = [wave0, wave1, wave2, wave3, wave4]
                for wave in waves:
                    s = recognizer.create_stream()
                    samples, sample_rate = read_wave(wave)
                    s.accept_waveform(sample_rate, samples)

                    tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                    s.accept_waveform(sample_rate, tail_paddings)
                    s.input_finished()
                    streams.append(s)

                while True:
                    ready_list = []
                    for s in streams:
                        if recognizer.is_ready(s):
                            ready_list.append(s)
                    if len(ready_list) == 0:
                        break
                    recognizer.decode_streams(ready_list)
                results = [recognizer.get_result(s) for s in streams]
                for wave_filename, result in zip(waves, results):
                    print(f"{wave_filename}\n{result}")
                    print("-" * 10)

    def test_zipformer2_ctc(self):
        m = "sherpa-mnn-streaming-zipformer-ctc-multi-zh-hans-2023-12-13"
        for use_int8 in [True, False]:
            name = (
                "ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx"
                if use_int8
                else "ctc-epoch-20-avg-1-chunk-16-left-128.onnx"
            )
            model = f"{d}/{m}/{name}"
            tokens = f"{d}/{m}/tokens.txt"
            wave0 = f"{d}/{m}/test_wavs/DEV_T0000000000.wav"
            wave1 = f"{d}/{m}/test_wavs/DEV_T0000000001.wav"
            wave2 = f"{d}/{m}/test_wavs/DEV_T0000000002.wav"
            if not Path(model).is_file():
                print("skipping test_zipformer2_ctc()")
                return
            print(f"testing {model}")

            recognizer = sherpa_mnn.OnlineRecognizer.from_zipformer2_ctc(
                model=model,
                tokens=tokens,
                num_threads=1,
                provider="cpu",
            )

            streams = []
            waves = [wave0, wave1, wave2]
            for wave in waves:
                s = recognizer.create_stream()
                samples, sample_rate = read_wave(wave)
                s.accept_waveform(sample_rate, samples)

                tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                s.accept_waveform(sample_rate, tail_paddings)
                s.input_finished()
                streams.append(s)

            while True:
                ready_list = []
                for s in streams:
                    if recognizer.is_ready(s):
                        ready_list.append(s)
                if len(ready_list) == 0:
                    break
                recognizer.decode_streams(ready_list)

            results = [recognizer.get_result(s) for s in streams]
            for wave_filename, result in zip(waves, results):
                print(f"{wave_filename}\n{result}")
                print("-" * 10)

    def test_wenet_ctc(self):
        models = [
            "sherpa-mnn-zh-wenet-aishell",
            "sherpa-mnn-zh-wenet-aishell2",
            "sherpa-mnn-zh-wenet-wenetspeech",
            "sherpa-mnn-zh-wenet-multi-cn",
            "sherpa-mnn-en-wenet-librispeech",
            "sherpa-mnn-en-wenet-gigaspeech",
        ]
        for m in models:
            for use_int8 in [True, False]:
                name = (
                    "model-streaming.int8.onnx" if use_int8 else "model-streaming.onnx"
                )
                model = f"{d}/{m}/{name}"
                tokens = f"{d}/{m}/tokens.txt"

                wave0 = f"{d}/{m}/test_wavs/0.wav"
                wave1 = f"{d}/{m}/test_wavs/1.wav"
                wave2 = f"{d}/{m}/test_wavs/8k.wav"

                if not Path(model).is_file():
                    print("skipping test_wenet_ctc()")
                    return

                recognizer = sherpa_mnn.OnlineRecognizer.from_wenet_ctc(
                    model=model,
                    tokens=tokens,
                    num_threads=1,
                    provider="cpu",
                )

                streams = []
                waves = [wave0, wave1, wave2]
                for wave in waves:
                    s = recognizer.create_stream()
                    samples, sample_rate = read_wave(wave)
                    s.accept_waveform(sample_rate, samples)

                    tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                    s.accept_waveform(sample_rate, tail_paddings)
                    s.input_finished()
                    streams.append(s)

                while True:
                    ready_list = []
                    for s in streams:
                        if recognizer.is_ready(s):
                            ready_list.append(s)
                    if len(ready_list) == 0:
                        break
                    recognizer.decode_streams(ready_list)

                results = [recognizer.get_result(s) for s in streams]
                for wave_filename, result in zip(waves, results):
                    print(f"{wave_filename}\n{result}")
                    print("-" * 10)


if __name__ == "__main__":
    unittest.main()
