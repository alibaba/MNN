#!/usr/bin/env python3

"""
This script shows how to use Python APIs for speaker identification with
a microphone and a VAD model

Usage:

(1) Download a model for computing speaker embeddings

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
to download a model. An example is given below:

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx

Note that `zh` means Chinese, while `en` means English.

(2) Download the VAD model
Please visit
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
to download silero_vad.onnx

For instance,

wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

(3) Run this script

python3 ./python-api-examples/speaker-identification-with-vad-dynamic.py \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --model ./3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx
"""
import argparse
import sys

import numpy as np
import sherpa_mnn

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

g_sample_rate = 16000


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the speaker embedding model file.",
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )

    parser.add_argument("--threshold", type=float, default=0.4)

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    return parser.parse_args()


def load_speaker_embedding_model(args):
    config = sherpa_mnn.SpeakerEmbeddingExtractorConfig(
        model=args.model,
        num_threads=args.num_threads,
        debug=args.debug,
        provider=args.provider,
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    extractor = sherpa_mnn.SpeakerEmbeddingExtractor(config)
    return extractor


def compute_speaker_embedding(
    samples: np.ndarray,
    extractor: sherpa_mnn.SpeakerEmbeddingExtractor,
) -> np.ndarray:
    """
    Args:
      samples:
        A 1-D float32 array.
      extractor:
        The return value of function load_speaker_embedding_model().
    Returns:
      Return a 1-D float32 array.
    """
    if len(samples) < g_sample_rate:
        print(f"Your input contains only {len(samples)} samples!")

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=g_sample_rate, waveform=samples)
    stream.input_finished()

    assert extractor.is_ready(stream)
    embedding = extractor.compute(stream)
    embedding = np.array(embedding)
    return embedding


def main():
    args = get_args()
    print(args)

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    # If you want to select a different device, please change
    # sd.default.device[0]. For instance, if you want to select device 10,
    # please use
    #
    #  sd.default.device[0] = 4
    #  print(devices)
    #

    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    extractor = load_speaker_embedding_model(args)

    manager = sherpa_mnn.SpeakerEmbeddingManager(extractor.dim)

    vad_config = sherpa_mnn.VadModelConfig()
    vad_config.silero_vad.model = args.silero_vad_model
    vad_config.silero_vad.min_silence_duration = 0.25
    vad_config.silero_vad.min_speech_duration = 1.0
    vad_config.sample_rate = g_sample_rate

    window_size = vad_config.silero_vad.window_size
    vad = sherpa_mnn.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    samples_per_read = int(0.1 * g_sample_rate)  # 0.1 second = 100 ms

    print("Started! Please speak")

    line_num = 0
    speaker_id = 0
    buffer = []
    with sd.InputStream(channels=1, dtype="float32", samplerate=g_sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            while not vad.empty():
                if len(vad.front.samples) < 0.5 * g_sample_rate:
                    # this segment is too short, skip it
                    vad.pop()
                    continue
                stream = extractor.create_stream()
                stream.accept_waveform(
                    sample_rate=g_sample_rate, waveform=vad.front.samples
                )
                vad.pop()
                stream.input_finished()

                embedding = extractor.compute(stream)
                embedding = np.array(embedding)
                name = manager.search(embedding, threshold=args.threshold)
                if not name:
                    # register it
                    new_name = f"speaker_{speaker_id}"
                    status = manager.add(new_name, embedding)
                    if not status:
                        raise RuntimeError(f"Failed to register speaker {new_name}")
                    print(
                        f"{line_num}: Detected new speaker. Register it as {new_name}"
                    )
                    speaker_id += 1
                else:
                    print(f"{line_num}: Detected existing speaker: {name}")
                line_num += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
