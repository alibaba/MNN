#!/usr/bin/env python3

"""
This script shows how to use Python APIs for speaker identification with
a microphone and a VAD model

Usage:

(1) Prepare a text file containing speaker related files.

Each line in the text file contains two columns. The first column is the
speaker name, while the second column contains the wave file of the speaker.

If the text file contains multiple wave files for the same speaker, then the
embeddings of these files are averaged.

An example text file is given below:

    foo /path/to/a.wav
    bar /path/to/b.wav
    foo /path/to/c.wav
    foobar /path/to/d.wav

Each wave file should contain only a single channel; the sample format
should be int16_t; the sample rate can be arbitrary.

(2) Download a model for computing speaker embeddings

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
to download a model. An example is given below:

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx

Note that `zh` means Chinese, while `en` means English.

(3) Download the VAD model
Please visit
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
to download silero_vad.onnx

For instance,

wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

(4) Run this script

Assume the filename of the text file is speaker.txt.

python3 ./python-api-examples/speaker-identification-with-vad.py \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --speaker-file ./speaker.txt \
  --model ./wespeaker_zh_cnceleb_resnet34.onnx
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--speaker-file",
        type=str,
        required=True,
        help="""Path to the speaker file. Read the help doc at the beginning of this
        file for the format.""",
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

    parser.add_argument("--threshold", type=float, default=0.6)

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


def load_speaker_file(args) -> Dict[str, List[str]]:
    if not Path(args.speaker_file).is_file():
        raise ValueError(f"--speaker-file {args.speaker_file} does not exist")

    ans = defaultdict(list)
    with open(args.speaker_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid line: {line}. Fields: {fields}")

            speaker_name, filename = fields
            ans[speaker_name].append(filename)
    return ans


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_speaker_embedding(
    filenames: List[str],
    extractor: sherpa_mnn.SpeakerEmbeddingExtractor,
) -> np.ndarray:
    assert len(filenames) > 0, "filenames is empty"

    ans = None
    for filename in filenames:
        print(f"processing {filename}")
        samples, sample_rate = load_audio(filename)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        if ans is None:
            ans = embedding
        else:
            ans += embedding

    return ans / len(filenames)


g_sample_rate = 16000


def main():
    args = get_args()
    print(args)
    extractor = load_speaker_embedding_model(args)
    speaker_file = load_speaker_file(args)

    manager = sherpa_mnn.SpeakerEmbeddingManager(extractor.dim)
    for name, filename_list in speaker_file.items():
        embedding = compute_speaker_embedding(
            filenames=filename_list,
            extractor=extractor,
        )
        status = manager.add(name, embedding)
        if not status:
            raise RuntimeError(f"Failed to register speaker {name}")

    vad_config = sherpa_mnn.VadModelConfig()
    vad_config.silero_vad.model = args.silero_vad_model
    vad_config.silero_vad.min_silence_duration = 0.25
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.sample_rate = g_sample_rate

    window_size = vad_config.silero_vad.window_size
    vad = sherpa_mnn.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    samples_per_read = int(0.1 * g_sample_rate)  # 0.1 second = 100 ms

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    print("Started! Please speak")

    idx = 0
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

                print("Computing", end="")
                embedding = extractor.compute(stream)
                embedding = np.array(embedding)
                name = manager.search(embedding, threshold=args.threshold)
                if not name:
                    name = "unknown"
                print(f"\r{idx}: Predicted name: {name}")
                idx += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
