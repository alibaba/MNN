#!/usr/bin/env python3

"""
This file shows how to use a non-streaming FireRedAsr AED model from
https://github.com/FireRedTeam/FireRedASR
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
"""

from pathlib import Path

import sherpa_mnn
import soundfile as sf


def create_recognizer():
    encoder = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx"
    decoder = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx"
    tokens = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt"
    test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav"
    #  test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/1.wav"
    #  test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/2.wav"
    #  test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/3.wav"
    #  test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/8k.wav"
    #  test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/3-sichuan.wav"
    #  test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/4-tianjin.wav"
    #  test_wav = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/5-henan.wav"

    if (
        not Path(encoder).is_file()
        or not Path(decoder).is_file()
        or not Path(test_wav).is_file()
    ):
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_mnn.OfflineRecognizer.from_fire_red_asr(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            debug=True,
        ),
        test_wav,
    )


def main():
    recognizer, wave_filename = create_recognizer()

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    # audio is a 1-D float32 numpy array normalized to the range [-1, 1]
    # sample_rate does not need to be 16000 Hz

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    print(wave_filename)
    print(stream.result)


if __name__ == "__main__":
    main()
