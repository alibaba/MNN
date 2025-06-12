#!/usr/bin/env python3

"""
This file shows how to use a non-streaming transducer model from NeMo
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models


The example model supports 10 languages and it is converted from
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_multilingual_fastconformer_hybrid_large_pc
"""

from pathlib import Path

import sherpa_mnn
import soundfile as sf


def create_recognizer():
    encoder = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/encoder.onnx"
    decoder = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/decoder.onnx"
    joiner = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/joiner.onnx"
    tokens = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/tokens.txt"

    test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/de-german.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/en-english.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/es-spanish.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/fr-french.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/hr-croatian.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/it-italian.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/po-polish.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/ru-russian.wav"
    #  test_wav = "./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/uk-ukrainian.wav"

    if not Path(encoder).is_file() or not Path(test_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_mnn.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            model_type="nemo_transducer",
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
