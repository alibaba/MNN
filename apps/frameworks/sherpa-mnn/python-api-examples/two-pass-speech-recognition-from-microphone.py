#!/usr/bin/env python3

# Two-pass real-time speech recognition from a microphone with sherpa-onnx
# Python API.
#
# The first pass uses a streaming model, which has two purposes:
#
#  (1) Display a temporary result to users
#
#  (2) Endpointing
#
# The second pass uses a non-streaming model. It has a higher recognition
# accuracy than the first pass model and its result is used as the final result.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

"""
Usage examples:

(1) Chinese: Streaming zipformer (1st pass) + Non-streaming paraformer (2nd pass)

python3 ./python-api-examples/two-pass-speech-recognition-from-microphone.py \
  --first-encoder ./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/encoder-epoch-99-avg-1.onnx \
  --first-decoder ./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/decoder-epoch-99-avg-1.onnx \
  --first-joiner ./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/joiner-epoch-99-avg-1.onnx \
  --first-tokens ./sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/tokens.txt \
  \
  --second-paraformer ./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
  --second-tokens ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt

(2) English: Streaming zipformer (1st pass) + Non-streaming whisper (2nd pass)

python3 ./python-api-examples/two-pass-speech-recognition-from-microphone.py \
  --first-encoder ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \
  --first-decoder ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \
  --first-joiner ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \
  --first-tokens ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \
  \
  --second-whisper-encoder ./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx \
  --second-whisper-decoder ./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx \
  --second-tokens ./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

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


def assert_file_exists(filename: str, message: str):
    if not filename:
        raise ValueError(f"Please specify {message}")

    if not Path(filename).is_file():
        raise ValueError(f"{message} {filename} does not exist")


def add_first_pass_streaming_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--first-tokens",
        type=str,
        required=True,
        help="Path to tokens.txt for the first pass",
    )

    parser.add_argument(
        "--first-encoder",
        type=str,
        required=True,
        help="Path to the encoder model for the first pass",
    )

    parser.add_argument(
        "--first-decoder",
        type=str,
        required=True,
        help="Path to the decoder model for the first pass",
    )

    parser.add_argument(
        "--first-joiner",
        type=str,
        help="Path to the joiner model for the first pass",
    )

    parser.add_argument(
        "--first-decoding-method",
        type=str,
        default="greedy_search",
        help="""Decoding method for the first pass. Valid values are
        greedy_search and modified_beam_search""",
    )

    parser.add_argument(
        "--first-max-active-paths",
        type=int,
        default=4,
        help="""Used only when --first-decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )


def add_second_pass_transducer_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--second-encoder",
        default="",
        type=str,
        help="Path to the transducer encoder model for the second pass",
    )

    parser.add_argument(
        "--second-decoder",
        default="",
        type=str,
        help="Path to the transducer decoder model for the second pass",
    )

    parser.add_argument(
        "--second-joiner",
        default="",
        type=str,
        help="Path to the transducer joiner model for the second pass",
    )


def add_second_pass_paraformer_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--second-paraformer",
        default="",
        type=str,
        help="Path to the model.onnx for Paraformer for the second pass",
    )


def add_second_pass_nemo_ctc_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--second-nemo-ctc",
        default="",
        type=str,
        help="Path to the model.onnx for NeMo CTC for the second pass",
    )


def add_second_pass_whisper_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--second-whisper-encoder",
        default="",
        type=str,
        help="Path to whisper encoder model for the second pass",
    )

    parser.add_argument(
        "--second-whisper-decoder",
        default="",
        type=str,
        help="Path to whisper decoder model for the second pass",
    )

    parser.add_argument(
        "--second-whisper-language",
        default="",
        type=str,
        help="""It specifies the spoken language in the input audio file.
        Example values: en, fr, de, zh, jp.
        Available languages for multilingual models can be found at
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        If not specified, we infer the language from the input audio file.
        """,
    )

    parser.add_argument(
        "--second-whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="""For multilingual models, if you specify translate, the output
        will be in English.
        """,
    )

    parser.add_argument(
        "--second-whisper-tail-paddings",
        default=-1,
        type=int,
        help="""Number of tail padding frames.
        We have removed the 30-second constraint from whisper, so you need to
        choose the amount of tail padding frames by yourself.
        Use -1 to use a default value for tail padding.
        """,
    )


def add_second_pass_non_streaming_model_args(parser: argparse.ArgumentParser):
    add_second_pass_transducer_model_args(parser)
    add_second_pass_nemo_ctc_model_args(parser)
    add_second_pass_paraformer_model_args(parser)
    add_second_pass_whisper_model_args(parser)

    parser.add_argument(
        "--second-tokens",
        type=str,
        help="Path to tokens.txt for the second pass",
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    add_first_pass_streaming_model_args(parser)
    add_second_pass_non_streaming_model_args(parser)

    return parser.parse_args()


def check_first_pass_args(args):
    assert_file_exists(args.first_tokens, "--first-tokens")
    assert_file_exists(args.first_encoder, "--first-encoder")
    assert_file_exists(args.first_decoder, "--first-decoder")
    assert_file_exists(args.first_joiner, "--first-joiner")


def check_second_pass_args(args):
    assert_file_exists(args.second_tokens, "--second-tokens")

    if args.second_encoder:
        assert_file_exists(args.second_encoder, "--second-encoder")
        assert_file_exists(args.second_decoder, "--second-decoder")
        assert_file_exists(args.second_joiner, "--second-joiner")
    elif args.second_paraformer:
        assert_file_exists(args.second_paraformer, "--second-paraformer")
    elif args.second_nemo_ctc:
        assert_file_exists(args.second_nemo_ctc, "--second-nemo-ctc")
    elif args.second_whisper_encoder:
        assert_file_exists(args.second_whisper_encoder, "--second-whisper-encoder")
        assert_file_exists(args.second_whisper_decoder, "--second-whisper-decoder")
    else:
        raise ValueError("Please specify the model for the second pass")


def create_first_pass_recognizer(args):
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_mnn.OnlineRecognizer.from_transducer(
        tokens=args.first_tokens,
        encoder=args.first_encoder,
        decoder=args.first_decoder,
        joiner=args.first_joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.first_decoding_method,
        max_active_paths=args.first_max_active_paths,
        provider=args.provider,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
    )
    return recognizer


def create_second_pass_recognizer(args) -> sherpa_mnn.OfflineRecognizer:
    if args.second_encoder:
        recognizer = sherpa_mnn.OfflineRecognizer.from_transducer(
            encoder=args.second_encoder,
            decoder=args.second_decoder,
            joiner=args.second_joiner,
            tokens=args.second_tokens,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            max_active_paths=4,
        )
    elif args.second_paraformer:
        recognizer = sherpa_mnn.OfflineRecognizer.from_paraformer(
            paraformer=args.second_paraformer,
            tokens=args.second_tokens,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    elif args.second_nemo_ctc:
        recognizer = sherpa_mnn.OfflineRecognizer.from_nemo_ctc(
            model=args.second_nemo_ctc,
            tokens=args.second_tokens,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    elif args.second_whisper_encoder:
        recognizer = sherpa_mnn.OfflineRecognizer.from_whisper(
            encoder=args.second_whisper_encoder,
            decoder=args.second_whisper_decoder,
            tokens=args.second_tokens,
            num_threads=1,
            decoding_method="greedy_search",
            language=args.second_whisper_language,
            task=args.second_whisper_task,
            tail_paddings=args.second_whisper_tail_paddings,
        )
    else:
        raise ValueError("Please specify at least one model for the second pass")

    return recognizer


def run_second_pass(
    recognizer: sherpa_mnn.OfflineRecognizer,
    samples: np.ndarray,
    sample_rate: int,
):
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    recognizer.decode_stream(stream)

    return stream.result.text


def main():
    args = get_args()
    check_first_pass_args(args)
    check_second_pass_args(args)

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)

    # If you want to select a different input device, please use
    # sd.default.device[0] = xxx
    # where xxx is the device number

    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    print("Creating recognizers. Please wait...")
    first_recognizer = create_first_pass_recognizer(args)
    second_recognizer = create_second_pass_recognizer(args)

    print("Started! Please speak")

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    stream = first_recognizer.create_stream()

    last_result = ""
    segment_id = 0

    sample_buffers = []
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)

            sample_buffers.append(samples)

            while first_recognizer.is_ready(stream):
                first_recognizer.decode_stream(stream)

            is_endpoint = first_recognizer.is_endpoint(stream)

            result = first_recognizer.get_result(stream)
            result = result.lower().strip()

            if last_result != result:
                print(
                    "\r{}:{}".format(segment_id, " " * len(last_result)),
                    end="",
                    flush=True,
                )
                last_result = result
                print("\r{}:{}".format(segment_id, result), end="", flush=True)

            if is_endpoint:
                if result:
                    samples = np.concatenate(sample_buffers)
                    # There are internal sample buffers inside the streaming
                    # feature extractor, so we cannot send all samples to
                    # the 2nd pass. Here 8000 is just an empirical value
                    # that should work for most streaming models in sherpa-onnx
                    sample_buffers = [samples[-8000:]]
                    samples = samples[:-8000]
                    result = run_second_pass(
                        recognizer=second_recognizer,
                        samples=samples,
                        sample_rate=sample_rate,
                    )
                    result = result.lower().strip()

                    print(
                        "\r{}:{}".format(segment_id, " " * len(last_result)),
                        end="",
                        flush=True,
                    )
                    print("\r{}:{}".format(segment_id, result), flush=True)
                    segment_id += 1
                else:
                    sample_buffers = []

                first_recognizer.reset(stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
