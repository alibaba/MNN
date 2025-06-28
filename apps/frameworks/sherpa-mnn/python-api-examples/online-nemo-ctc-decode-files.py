#!/usr/bin/env python3

"""
This file shows how to use a streaming CTC model from NeMo
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models


The example model is converted from
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms
"""

from pathlib import Path

import numpy as np
import sherpa_mnn
import soundfile as sf


def create_recognizer():
    model = "./sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/model.onnx"
    tokens = "./sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/tokens.txt"

    test_wav = "./sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/test_wavs/0.wav"

    if not Path(model).is_file() or not Path(test_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_mnn.OnlineRecognizer.from_nemo_ctc(
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

    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    print(wave_filename)
    print(recognizer.get_result_all(stream))


if __name__ == "__main__":
    main()
