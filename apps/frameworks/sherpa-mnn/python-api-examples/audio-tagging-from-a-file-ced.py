#!/usr/bin/env python3

"""
This script shows how to use audio tagging Python APIs to tag a file.

Please read the code to download the required model files and test wave file.
"""

import logging
import time
from pathlib import Path

import numpy as np
import sherpa_mnn
import soundfile as sf


def read_test_wave():
    # Please download the model files and test wave files from
    # https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
    test_wave = "./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/test_wavs/6.wav"

    if not Path(test_wave).is_file():
        raise ValueError(
            f"Please download {test_wave} from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    # See https://python-soundfile.readthedocs.io/en/0.11.0/#soundfile.read
    data, sample_rate = sf.read(
        test_wave,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)

    # samples is a 1-d array of dtype float32
    # sample_rate is a scalar
    return samples, sample_rate


def create_audio_tagger():
    # Please download the model files and test wave files from
    # https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
    model_file = "./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx"
    label_file = (
        "./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/class_labels_indices.csv"
    )

    if not Path(model_file).is_file():
        raise ValueError(
            f"Please download {model_file} from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    if not Path(label_file).is_file():
        raise ValueError(
            f"Please download {label_file} from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )

    config = sherpa_mnn.AudioTaggingConfig(
        model=sherpa_mnn.AudioTaggingModelConfig(
            ced=model_file,
            num_threads=1,
            debug=True,
            provider="cpu",
        ),
        labels=label_file,
        top_k=5,
    )
    if not config.validate():
        raise ValueError(f"Please check the config: {config}")

    print(config)

    return sherpa_mnn.AudioTagging(config)


def main():
    logging.info("Create audio tagger")
    audio_tagger = create_audio_tagger()

    logging.info("Read test wave")
    samples, sample_rate = read_test_wave()

    logging.info("Computing")

    start_time = time.time()

    stream = audio_tagger.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    result = audio_tagger.compute(stream)
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    audio_duration = len(samples) / sample_rate

    real_time_factor = elapsed_seconds / audio_duration
    logging.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
    logging.info(f"Audio duration in seconds: {audio_duration:.3f}")
    logging.info(
        f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    s = "\n"
    for i, e in enumerate(result):
        s += f"{i}: {e}\n"

    logging.info(s)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
