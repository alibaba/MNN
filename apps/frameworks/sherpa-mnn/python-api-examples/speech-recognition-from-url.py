#!/usr/bin/env python3
#
# Real-time speech recognition from a URL with sherpa-onnx Python API
#
# Supported URLs are those supported by ffmpeg.
#
# For instance:
# (1) RTMP
#     rtmp://localhost/live/livestream
#
# (2) A file
#     https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/wenetspeech/DEV_T0000000000.opus
#     https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/aishell2/ID0012W0030.wav
#     file:///Users/fangjun/open-source/sherpa-onnx/a.wav
#
#    Note that it supports all file formats supported by ffmpeg
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import sherpa_mnn


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
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="""Example values:
          rtmp://localhost/live/livestream
          https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/wenetspeech/DEV_T0000000000.opus
          https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/aishell2/ID0012W0030.wav
        """,
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, and for each
        phrase the bpe/cjkchar are separated by a space. For example:

        ▁HE LL O ▁WORLD
        你 好 世 界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )


    return parser.parse_args()


def create_recognizer(args):
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_mnn.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,  # it essentially disables this rule
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
    )
    return recognizer


def main():
    args = get_args()
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)

    recognizer = create_recognizer(args)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        args.url,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    frames_per_read = 1600  # 0.1 second

    stream = recognizer.create_stream()

    last_result = ""
    segment_id = 0

    print("Started!")
    while True:
        # *2 because int16_t has two bytes
        data = process.stdout.read(frames_per_read * 2)
        if not data:
            break

        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768
        stream.accept_waveform(16000, samples)

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
    if shutil.which("ffmpeg") is None:
        sys.exit("Please install ffmpeg first!")
    main()
