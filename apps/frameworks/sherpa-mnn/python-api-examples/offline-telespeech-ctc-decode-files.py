#!/usr/bin/env python3

"""
This file shows how to use a non-streaming CTC model from
https://github.com/Tele-AI/TeleSpeech-ASR
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models


"""

from pathlib import Path

import sherpa_mnn
import soundfile as sf


def create_recognizer():
    model = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/model.int8.onnx"
    tokens = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt"
    test_wav = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/3-sichuan.wav"
    #  test_wav = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/4-tianjin.wav"
    #  test_wav = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/5-henan.wav"

    if not Path(model).is_file() or not Path(test_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_mnn.OfflineRecognizer.from_telespeech_ctc(
            model=model,
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
