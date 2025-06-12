#!/usr/bin/env python3

"""
This file shows how to use the speech enhancement API.

Please download files used this script from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

Example:

 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_mnn
import soundfile as sf


def create_speech_denoiser():
    model_filename = "./gtcrn_simple.onnx"
    if not Path(model_filename).is_file():
        raise ValueError(
            "Please first download a model from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    config = sherpa_mnn.OfflineSpeechDenoiserConfig(
        model=sherpa_mnn.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_mnn.OfflineSpeechDenoiserGtcrnModelConfig(
                model=model_filename
            ),
            debug=False,
            num_threads=1,
            provider="cpu",
        )
    )
    if not config.validate():
        print(config)
        raise ValueError("Errors in config. Please check previous error logs")
    return sherpa_mnn.OfflineSpeechDenoiser(config)


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def main():
    sd = create_speech_denoiser()
    test_wave = "./speech_with_noise.wav"
    if not Path(test_wave).is_file():
        raise ValueError(
            f"{test_wave} does not exist. You can download it from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    samples, sample_rate = load_audio(test_wave)

    start = time.time()
    denoised = sd(samples, sample_rate)
    end = time.time()

    elapsed_seconds = end - start
    audio_duration = len(samples) / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write("./enhanced_16k.wav", denoised.samples, denoised.sample_rate)
    print("Saved to ./enhanced_16k.wav")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
