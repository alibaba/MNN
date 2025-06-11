#!/usr/bin/env python3

"""
This script works only on Linux. It uses ALSA for recording.
"""

import argparse
from pathlib import Path

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

    parser.add_argument(
        "--device-name",
        type=str,
        required=True,
        help="""
The device name specifies which microphone to use in case there are several
on your system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and device 0 on that card, please use:

  plughw:3,0

as the device_name.
        """,
    )

    return parser.parse_args()


def main():
    args = get_args()
    if not Path(args.silero_vad_model).is_file():
        raise RuntimeError(
            f"{args.silero_vad_model} does not exist. Please download it from "
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        )

    device_name = args.device_name
    print(f"device_name: {device_name}")
    alsa = sherpa_mnn.Alsa(device_name)

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    config = sherpa_mnn.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.sample_rate = sample_rate

    vad = sherpa_mnn.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    print("Started! Please speak. Press Ctrl C to exit")

    printed = False
    k = 0
    try:
        while True:
            samples = alsa.read(samples_per_read)  # a blocking read

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
