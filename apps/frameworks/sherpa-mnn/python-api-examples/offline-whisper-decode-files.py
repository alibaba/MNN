#!/usr/bin/env python3

"""
This file shows how to use a non-streaming whisper model from
https://github.com/openai/whisper
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2
"""

import datetime as dt
from pathlib import Path

import sherpa_mnn
import soundfile as sf


def create_recognizer():
    encoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx"
    decoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx"
    tokens = "./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt"
    test_wav = "./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav"

    if not Path(encoder).is_file() or not Path(test_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_mnn.OfflineRecognizer.from_whisper(
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

    start_t = dt.datetime.now()

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)

    end_t = dt.datetime.now()
    elapsed_seconds = (end_t - start_t).total_seconds()
    duration = audio.shape[-1] / sample_rate
    rtf = elapsed_seconds / duration

    print(stream.result)
    print(wave_filename)
    print("Text:", stream.result.text)
    print(f"Audio duration:\t{duration:.3f} s")
    print(f"Elapsed:\t{elapsed_seconds:.3f} s")
    print(f"RTF = {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")


if __name__ == "__main__":
    main()
