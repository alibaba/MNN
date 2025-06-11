#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
# with endpoint detection.
# This script uses a streaming paraformer
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#
# to download pre-trained models

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


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html to download it"
    )


def create_recognizer():
    encoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx"
    decoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx"
    tokens = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt"
    assert_file_exists(encoder)
    assert_file_exists(decoder)
    assert_file_exists(tokens)
    recognizer = sherpa_mnn.OnlineRecognizer.from_paraformer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,  # it essentially disables this rule
    )
    return recognizer


def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    recognizer = create_recognizer()
    print("Started! Please speak")

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    stream = recognizer.create_stream()

    last_result = ""
    segment_id = 0
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            is_endpoint = recognizer.is_endpoint(stream)

            result = recognizer.get_result(stream)

            if result and (last_result != result):
                last_result = result
                print("\r{}:{}".format(segment_id, result), end="", flush=True)
            if is_endpoint:
                if result:
                    print("\r{}:{}".format(segment_id, result), flush=True)
                    segment_id += 1
                recognizer.reset(stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
