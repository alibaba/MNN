#!/usr/bin/env python3

# Real-time keyword spotting from a microphone with sherpa-onnx Python API
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys
from pathlib import Path

from typing import List

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
        "https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html to download it"
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""
        It specifies number of active paths to keep during decoding.
        """,
    )

    parser.add_argument(
        "--num-trailing-blanks",
        type=int,
        default=1,
        help="""The number of trailing blanks a keyword should be followed. Setting
        to a larger value (e.g. 8) when your keywords has overlapping tokens
        between each other.
        """,
    )

    parser.add_argument(
        "--keywords-file",
        type=str,
        help="""
        The file containing keywords, one words/phrases per line, and for each
        phrase the bpe/cjkchar/pinyin are separated by a space. For example:

        ▁HE LL O ▁WORLD
        x iǎo ài t óng x ué 
        """,
    )

    parser.add_argument(
        "--keywords-score",
        type=float,
        default=1.0,
        help="""
        The boosting score of each token for keywords. The larger the easier to
        survive beam search.
        """,
    )

    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.25,
        help="""
        The trigger threshold (i.e. probability) of the keyword. The larger the
        harder to trigger.
        """,
    )

    return parser.parse_args()


def main():
    args = get_args()

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    assert_file_exists(args.tokens)
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)

    assert Path(
        args.keywords_file
    ).is_file(), (
        f"keywords_file : {args.keywords_file} not exist, please provide a valid path."
    )

    keyword_spotter = sherpa_mnn.KeywordSpotter(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        max_active_paths=args.max_active_paths,
        keywords_file=args.keywords_file,
        keywords_score=args.keywords_score,
        keywords_threshold=args.keywords_threshold,
        num_trailing_blanks=args.num_trailing_blanks,
        provider=args.provider,
    )

    print("Started! Please speak")

    idx = 0

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    stream = keyword_spotter.create_stream()
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while keyword_spotter.is_ready(stream):
                keyword_spotter.decode_stream(stream)
                result = keyword_spotter.get_result(stream)
                if result:
                    print(f"{idx}: {result }")
                    idx += 1
                    # Remember to reset stream right after detecting a keyword
                    keyword_spotter.reset_stream(stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
