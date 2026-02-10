#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python APIs to generate
subtitles.

Supported file formats are those supported by ffmpeg; for instance,
*.mov, *.mp4, *.wav, etc.

Note that you need a non-streaming model for this script.

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
to download silero_vad.onnx

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

(1) For paraformer

    ./python-api-examples/generate-subtitles.py  \
      --silero-vad-model=/path/to/silero_vad.onnx \
      --tokens=/path/to/tokens.txt \
      --paraformer=/path/to/paraformer.onnx \
      --num-threads=2 \
      --decoding-method=greedy_search \
      --debug=false \
      --sample-rate=16000 \
      --feature-dim=80 \
      /path/to/test.mp4

(2) For transducer models from icefall

    ./python-api-examples/generate-subtitles.py  \
      --silero-vad-model=/path/to/silero_vad.onnx \
      --tokens=/path/to/tokens.txt \
      --encoder=/path/to/encoder.onnx \
      --decoder=/path/to/decoder.onnx \
      --joiner=/path/to/joiner.onnx \
      --num-threads=2 \
      --decoding-method=greedy_search \
      --debug=false \
      --sample-rate=16000 \
      --feature-dim=80 \
      /path/to/test.mp4

(3) For Moonshine models

./python-api-examples/generate-subtitles.py  \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --moonshine-preprocessor=./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx \
  --moonshine-encoder=./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx \
  --moonshine-uncached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx \
  --moonshine-cached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx \
  --tokens=./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt \
  --num-threads=2 \
  /path/to/test.mp4

(4) For Whisper models

./python-api-examples/generate-subtitles.py  \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --whisper-encoder=./sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx \
  --tokens=./sherpa-onnx-whisper-base.en/base.en-tokens.txt \
  --whisper-task=transcribe \
  --num-threads=2 \
  /path/to/test.mp4

(5) For SenseVoice CTC models

./python-api-examples/generate-subtitles.py  \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
  --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --num-threads=2 \
  /path/to/test.mp4


(6) For WeNet CTC models

./python-api-examples/generate-subtitles.py  \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --wenet-ctc=./sherpa-onnx-zh-wenet-wenetspeech/model.onnx \
  --tokens=./sherpa-onnx-zh-wenet-wenetspeech/tokens.txt \
  --num-threads=2 \
  /path/to/test.mp4

Please refer to
https://k2-fsa.github.io/sherpa/onnx/index.html
to install sherpa-onnx and to download non-streaming pre-trained models
used in this file.
"""
import argparse
import datetime as dt
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
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
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to the model.onnx from SenseVoice",
    )

    parser.add_argument(
        "--wenet-ctc",
        default="",
        type=str,
        help="Path to the CTC model.onnx from WeNet",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--whisper-encoder",
        default="",
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default="",
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="""It specifies the spoken language in the input file.
        Example values: en, fr, de, zh, jp.
        Available languages for multilingual models can be found at
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        If not specified, we infer the language from the input audio file.
        """,
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="""For multilingual models, if you specify translate, the output
        will be in English.
        """,
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="""Number of tail padding frames.
        We have removed the 30-second constraint from whisper, so you need to
        choose the amount of tail padding frames by yourself.
        Use -1 to use a default value for tail padding.
        """,
    )

    parser.add_argument(
        "--moonshine-preprocessor",
        default="",
        type=str,
        help="Path to moonshine preprocessor model",
    )

    parser.add_argument(
        "--moonshine-encoder",
        default="",
        type=str,
        help="Path to moonshine encoder model",
    )

    parser.add_argument(
        "--moonshine-uncached-decoder",
        default="",
        type=str,
        help="Path to moonshine uncached decoder model",
    )

    parser.add_argument(
        "--moonshine-cached-decoder",
        default="",
        type=str,
        help="Path to moonshine cached decoder model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Valid values are greedy_search and modified_beam_search.
        modified_beam_search is valid only for transducer models.
        """,
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages when loading modes.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="""Sample rate of the feature extractor. Must match the one
        expected by the model. Note: The input sound files can have a
        different sample rate from this argument.""",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="The input sound file to generate subtitles ",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args) -> sherpa_mnn.OfflineRecognizer:
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.sense_voice) == 0, args.sense_voice
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        recognizer = sherpa_mnn.OfflineRecognizer.from_transducer(
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.paraformer:
        assert len(args.sense_voice) == 0, args.sense_voice
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.paraformer)

        recognizer = sherpa_mnn.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.sense_voice:
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.sense_voice)
        recognizer = sherpa_mnn.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice,
            tokens=args.tokens,
            num_threads=args.num_threads,
            use_itn=True,
            debug=args.debug,
        )
    elif args.wenet_ctc:
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.wenet_ctc)

        recognizer = sherpa_mnn.OfflineRecognizer.from_wenet_ctc(
            model=args.wenet_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.whisper_encoder:
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        recognizer = sherpa_mnn.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            debug=args.debug,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
        )
    elif args.moonshine_preprocessor:
        assert_file_exists(args.moonshine_preprocessor)
        assert_file_exists(args.moonshine_encoder)
        assert_file_exists(args.moonshine_uncached_decoder)
        assert_file_exists(args.moonshine_cached_decoder)

        recognizer = sherpa_mnn.OfflineRecognizer.from_moonshine(
            preprocessor=args.moonshine_preprocessor,
            encoder=args.moonshine_encoder,
            uncached_decoder=args.moonshine_uncached_decoder,
            cached_decoder=args.moonshine_cached_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    else:
        raise ValueError("Please specify at least one model")

    return recognizer


@dataclass
class Segment:
    start: float
    duration: float
    text: str = ""

    @property
    def end(self):
        return self.start + self.duration

    def __str__(self):
        s = f"{timedelta(seconds=self.start)}"[:-3]
        s += " --> "
        s += f"{timedelta(seconds=self.end)}"[:-3]
        s = s.replace(".", ",")
        s += "\n"
        s += self.text
        return s


def main():
    args = get_args()
    assert_file_exists(args.tokens)
    assert_file_exists(args.silero_vad_model)

    assert args.num_threads > 0, args.num_threads

    if not Path(args.sound_file).is_file():
        raise ValueError(f"{args.sound_file} does not exist")

    assert (
        args.sample_rate == 16000
    ), f"Only sample rate 16000 is supported.Given: {args.sample_rate}"

    recognizer = create_recognizer(args)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        args.sound_file,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(args.sample_rate),
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    frames_per_read = int(args.sample_rate * 100)  # 100 second

    stream = recognizer.create_stream()

    config = sherpa_mnn.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.25  # seconds
    config.silero_vad.min_speech_duration = 0.25  # seconds

    # If the current segment is larger than this value, then it increases
    # the threshold to 0.9 internally. After detecting this segment,
    # it resets the threshold to its original value.
    config.silero_vad.max_speech_duration = 5  # seconds
    config.sample_rate = args.sample_rate

    window_size = config.silero_vad.window_size

    buffer = []
    vad = sherpa_mnn.VoiceActivityDetector(config, buffer_size_in_seconds=100)

    segment_list = []

    print("Started!")
    start_t = dt.datetime.now()
    num_processed_samples = 0

    is_eof = False
    # TODO(fangjun): Support multithreads
    while not is_eof:
        # *2 because int16_t has two bytes
        data = process.stdout.read(frames_per_read * 2)
        if not data:
            vad.flush()
            is_eof = True
        else:
            samples = np.frombuffer(data, dtype=np.int16)
            samples = samples.astype(np.float32) / 32768

            num_processed_samples += samples.shape[0]

            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

        streams = []
        segments = []
        while not vad.empty():
            segment = Segment(
                start=vad.front.start / args.sample_rate,
                duration=len(vad.front.samples) / args.sample_rate,
            )
            segments.append(segment)

            stream = recognizer.create_stream()
            stream.accept_waveform(args.sample_rate, vad.front.samples)

            streams.append(stream)

            vad.pop()

        for s in streams:
            recognizer.decode_stream(s)

        for seg, stream in zip(segments, streams):
            seg.text = stream.result.text
            segment_list.append(seg)

    end_t = dt.datetime.now()
    elapsed_seconds = (end_t - start_t).total_seconds()
    duration = num_processed_samples / 16000
    rtf = elapsed_seconds / duration

    srt_filename = Path(args.sound_file).with_suffix(".srt")
    with open(srt_filename, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segment_list):
            print(i + 1, file=f)
            print(seg, file=f)
            print("", file=f)

    print(f"Saved to {srt_filename}")
    print(f"Audio duration:\t{duration:.3f} s")
    print(f"Elapsed:\t{elapsed_seconds:.3f} s")
    print(f"RTF = {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")
    print("Done!")


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        sys.exit("Please install ffmpeg first!")
    main()
