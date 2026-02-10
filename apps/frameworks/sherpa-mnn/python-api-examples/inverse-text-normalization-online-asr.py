#!/usr/bin/env python3
#
# Copyright (c)  2024  Xiaomi Corporation

"""
This script shows how to use inverse text normalization with streaming ASR.

Usage:

(1) Download the test model

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

(2) Download rule fst

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst

Please refer to
https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/itn_zh_number.ipynb
for how itn_zh_number.fst is generated.

(3) Download test wave

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

(4) Run this script

python3 ./python-api-examples/inverse-text-normalization-online-asr.py
"""
from pathlib import Path

import sherpa_mnn
import soundfile as sf


def create_recognizer():
    encoder = "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx"
    decoder = "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
    joiner = "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx"
    tokens = "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
    rule_fsts = "./itn_zh_number.fst"

    if (
        not Path(encoder).is_file()
        or not Path(decoder).is_file()
        or not Path(joiner).is_file()
        or not Path(tokens).is_file()
        or not Path(rule_fsts).is_file()
    ):
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return sherpa_mnn.OnlineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        debug=True,
        rule_fsts=rule_fsts,
    )


def main():
    recognizer = create_recognizer()
    wave_filename = "./itn-zh-number.wav"
    if not Path(wave_filename).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)

    tail_padding = [0] * int(0.3 * sample_rate)
    stream.accept_waveform(sample_rate, tail_padding)

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    print(wave_filename)
    print(recognizer.get_result_all(stream))


if __name__ == "__main__":
    main()
