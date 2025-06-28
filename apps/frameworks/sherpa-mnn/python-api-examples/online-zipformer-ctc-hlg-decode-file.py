#!/usr/bin/env python3

# This file shows how to use a streaming zipformer CTC model and an HLG
# graph for decoding.
#
# We use the following model as an example
#
"""
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2

python3 ./python-api-examples/online-zipformer-ctc-hlg-decode-file.py \
  --tokens ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt \
  --graph ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst \
  --model ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
  ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/0.wav

"""
# (The above model is from https://github.com/k2-fsa/icefall/pull/1557)

import argparse
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sherpa_mnn


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model",
    )

    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Path to H.fst, HL.fst, or HLG.fst",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Valid values: 1, 0",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="The input sound file to decode. It must be of WAVE"
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
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
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
    print(vars(args))

    assert_file_exists(args.tokens)
    assert_file_exists(args.graph)
    assert_file_exists(args.model)

    recognizer = sherpa_mnn.OnlineRecognizer.from_zipformer2_ctc(
        tokens=args.tokens,
        model=args.model,
        num_threads=args.num_threads,
        provider=args.provider,
        sample_rate=16000,
        feature_dim=80,
        ctc_graph=args.graph,
    )

    wave_filename = args.sound_file
    assert_file_exists(wave_filename)
    samples, sample_rate = read_wave(wave_filename)
    duration = len(samples) / sample_rate

    print("Started")

    start_time = time.time()
    s = recognizer.create_stream()
    s.accept_waveform(sample_rate, samples)
    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()
    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    result = recognizer.get_result(s).lower()
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / duration
    print(f"num_threads: {args.num_threads}")
    print(f"Wave duration: {duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(f"Real time factor (RTF): {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")
    print(result)


if __name__ == "__main__":
    main()
