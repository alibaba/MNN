#!/usr/bin/env python3

"""
This file shows how to remove non-speech segments
and merge all speech segments into a large segment
and save it to a file.

Usage

python3 ./vad-remove-non-speech-segments.py \
        --silero-vad-model silero_vad.onnx

Please visit
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
to download silero_vad.onnx

For instance,

wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_mnn
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )

    return parser.parse_args()


def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        print(
            "If you are using Linux and you are sure there is a microphone "
            "on your system, please use "
            "./vad-remove-non-speech-segments-alsa.py"
        )
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    args = get_args()
    assert_file_exists(args.silero_vad_model)

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    config = sherpa_mnn.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.sample_rate = sample_rate

    window_size = config.silero_vad.window_size

    buffer = []
    vad = sherpa_mnn.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    all_samples = []

    print("Started! Please speak. Press Ctrl C to exit")

    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)  # a blocking read
                samples = samples.reshape(-1)
                buffer = np.concatenate([buffer, samples])

                all_samples = np.concatenate([all_samples, samples])

                while len(buffer) > window_size:
                    vad.accept_waveform(buffer[:window_size])
                    buffer = buffer[window_size:]
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Saving & Exiting")

        speech_samples = []
        while not vad.empty():
            speech_samples.extend(vad.front.samples)
            vad.pop()

        speech_samples = np.array(speech_samples, dtype=np.float32)

        filename_for_speech = time.strftime("%Y%m%d-%H%M%S-speech.wav")
        sf.write(filename_for_speech, speech_samples, samplerate=sample_rate)

        filename_for_all = time.strftime("%Y%m%d-%H%M%S-all.wav")
        sf.write(filename_for_all, all_samples, samplerate=sample_rate)

        print(f"Saved to {filename_for_speech} and {filename_for_all}")


if __name__ == "__main__":
    main()
