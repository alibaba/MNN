#!/usr/bin/env python3

"""
This file shows how to use a non-streaming Moonshine model from
https://github.com/usefulsensors/moonshine
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
"""

import datetime as dt
from pathlib import Path

import sherpa_mnn
import soundfile as sf


def create_recognizer():
    preprocessor = "./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx"
    encoder = "./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx"
    uncached_decoder = "./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx"
    cached_decoder = "./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx"

    tokens = "./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt"
    test_wav = "./sherpa-onnx-moonshine-tiny-en-int8/test_wavs/0.wav"

    if not Path(preprocessor).is_file() or not Path(test_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_mnn.OfflineRecognizer.from_moonshine(
            preprocessor=preprocessor,
            encoder=encoder,
            uncached_decoder=uncached_decoder,
            cached_decoder=cached_decoder,
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
