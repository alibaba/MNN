#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_mnn


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
    args = get_args()
    if not Path(args.silero_vad_model).is_file():
        raise RuntimeError(
            f"{args.silero_vad_model} does not exist. Please download it from "
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        )

    mic_sample_rate = 16000
    if "SHERPA_ONNX_MIC_SAMPLE_RATE" in os.environ:
        mic_sample_rate = int(os.environ.get("SHERPA_ONNX_MIC_SAMPLE_RATE"))
        print(f"Change microphone sample rate to {mic_sample_rate}")

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    config = sherpa_mnn.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.sample_rate = sample_rate

    vad = sherpa_mnn.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    # python3 -m sounddevice
    # can also be used to list all devices

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        print(
            "If you are using Linux and you are sure there is a microphone "
            "on your system, please use "
            "./vad-alsa.py"
        )
        sys.exit(0)

    print(devices)

    if "SHERPA_ONNX_MIC_DEVICE" in os.environ:
        input_device_idx = int(os.environ.get("SHERPA_ONNX_MIC_DEVICE"))
        sd.default.device[0] = input_device_idx
        print(f'Use selected device: {devices[input_device_idx]["name"]}')
    else:
        input_device_idx = sd.default.device[0]
        print(f'Use default device: {devices[input_device_idx]["name"]}')

    print("Started! Please speak. Press Ctrl C to exit")

    printed = False
    k = 0
    try:
        with sd.InputStream(
            channels=1, dtype="float32", samplerate=mic_sample_rate
        ) as s:
            while True:
                samples, _ = s.read(samples_per_read)  # a blocking read
                samples = samples.reshape(-1)

                if mic_sample_rate != sample_rate:
                    import librosa

                    samples = librosa.resample(
                        samples, orig_sr=mic_sample_rate, target_sr=sample_rate
                    )

                vad.accept_waveform(samples)

                if vad.is_speech_detected() and not printed:
                    print("Detected speech")
                    printed = True

                if not vad.is_speech_detected():
                    printed = False

                while not vad.empty():
                    samples = vad.front.samples
                    duration = len(samples) / sample_rate
                    filename = f"seg-{k}-{duration:.3f}-seconds.wav"
                    k += 1
                    sherpa_mnn.write_wave(filename, samples, sample_rate)
                    print(f"Duration: {duration:.3f} seconds")
                    print(f"Saved to {filename}")
                    print("----------")

                    vad.pop()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exit")


if __name__ == "__main__":
    main()
